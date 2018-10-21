import torch
import torch.utils.data as D
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
import os.path
import random
from collections import OrderedDict
import skimage
from enum import Enum
from math import ceil

NUM_ACTIONS = 5
PHI_SIZE = 32*3*3
LEARNING_RATE = 1e-3
BETA = 0.2
BATCH_SIZE = 4
SKIP_DECAY_CONSTANT = 0.85**BATCH_SIZE
TEST_RATE_PCT = 0.1
LENS_SIZE = 33
IMG_COVG_PCT = 1.25


def to_phi_input(s):
    return to_tensor_f(s).view(BATCH_SIZE, 3, LENS_SIZE, -1)


def to_tensor_f(l):
    return to_tensor(l, type=torch.FloatTensor)


def to_tensor(l, type=torch.LongTensor):
    return type(l)


class Action(Enum):
    UP = [-1, 0]
    RIGHT = [0, 1]
    DOWN = [1, 0]
    LEFT = [0, -1]
    NEXT = None

    def to_onehot(self):
        onehot = torch.zeros([BATCH_SIZE, NUM_ACTIONS], dtype=torch.long)
        # note that this assumes same action for all training points in minibatch
        for i, a in enumerate(Action):
            if self == list(Action)[i]:
                onehot[:, i] = 1
        return onehot


class CropToMultiple(object):
    def __init__(self, m):
        self.m = m

    def __call__(self, image):
        h, w = image.shape[:2]
        return image[:((h//self.m)*self.m), :((w//self.m)*self.m)]


class Resize(object):
    # Adapted with permission from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        return skimage.transform.resize(image, (new_h, new_w))


class SUNDataset(D.Dataset):  # TODO
    def __init__(self, path, transforms=[], train=True):
        self.path = path
        self.transforms = transforms
        # self.files_accessed = set()
        self.files = sorted([os.path.join(self.path, name) for name in os.listdir(
            self.path) if os.path.isfile(os.path.join(self.path, name)) and name.endswith('.jpg')])
        if train:
            self.files = self.files[:int(len(self.files) * (1-TEST_RATE_PCT))]
        else:
            self.files = self.files[int(len(self.files) * (1-TEST_RATE_PCT)):]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = skimage.io.imread(self.files[i])
        for t in self.transforms:
            img = t(img)
        # self.files_accessed.add(self.files[i])
        return img

    def display_image(self, i):
        plt.figure()
        plt.imshow(self[i])
        plt.show()

    @staticmethod
    def collate(batch):
        return batch

    # def done(self):
    #     return len(self.files_accessed) == len(self)


class FeatureMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(OrderedDict(
            [
                ('conv1', nn.Conv2d(3, 32, 3, stride=2, padding=1)),
                ('elu1', nn.ELU()),
                ('conv2', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
                ('elu2', nn.ELU()),
                ('conv3', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
                ('elu3', nn.ELU()),
                ('conv4', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
                ('elu4', nn.ELU()),
            ]
        ))

    def forward(self, x):
        return self.flatten(self.net(x))

    def flatten(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)


class ActionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = FeatureMapper()
        # the embeddings of s_t and s_{t+1} are concatenated. each embedding has dimension 32*3*3=288
        self.fc1 = nn.Linear(2*PHI_SIZE, 256)
        self.fc2 = nn.Linear(256, NUM_ACTIONS)

    def forward(self, s_t0, s_t1):
        phi_cat = torch.cat((self.phi(to_phi_input(s_t0)),
                             self.phi(to_phi_input(s_t1))), 1)  # concat the two state vectors as input to the mlp
        return self.fc2(self.fc1(phi_cat))


class StateFeaturePredictor(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.fc1 = nn.Linear(PHI_SIZE+NUM_ACTIONS, 256)
        self.fc2 = nn.Linear(256, PHI_SIZE)
        self.phi = phi

    def forward(self, s_t0, a_t0):
        with torch.no_grad():
            # TODO does setting no_grad stop phi from being affected in SFP backprop?
            a_t0_onehot = a_t0.to_onehot()
            v = torch.cat((self.phi(to_phi_input(s_t0)), a_t0_onehot), 1)
        return self.fc2(self.fc1(v))


class ActionPolicy():
    def __init__(self, lens_dims):
        self.remaining = to_tensor(
            [int(l[0]*l[1]*IMG_COVG_PCT) for l in lens_dims])

    def act(self):
        for r in self.remaining:
            if SKIP_DECAY_CONSTANT**(r.item()) >= random.random():
                return Action.NEXT
        self.remaining -= 1
        return random.choice(list(Action)[:-1])


class ActionEnvironment():
    def __init__(self, batch):
        self.batch = batch
        # in units of LENS_SIZE
        self.coords = torch.zeros([BATCH_SIZE, 2], dtype=torch.long)
        self.dims = to_tensor(
            [[ceil(im.shape[0]/LENS_SIZE), ceil(im.shape[1]/LENS_SIZE)] for im in self.batch])
        self.update_state()
        self.policy = ActionPolicy(self.dims)
        self.done = False
        self.last_action = None

    def update_state(self):
        self.state = [im[int(co[0]*LENS_SIZE):int((co[0]+1)*LENS_SIZE), int(co[1]*LENS_SIZE):int((co[1]+1)*LENS_SIZE)]
                      for im, co in zip(self.batch, self.coords)]
        # overflow results in non square lens

    def step(self):
        self.last_action = self.policy.act()
        if self.last_action != Action.NEXT:
            self.coords += to_tensor(self.last_action.value)
            # handle wraparound in case of dimension overflow
            self.coords = torch.remainder(self.coords, self.dims)
            self.update_state()
        else:
            self.done = True


def loss_fn(a_hat, a, phi_hat, phi):
    fwd_loss = F.mse_loss(phi_hat, phi)
    inv_loss = F.cross_entropy(a_hat, a.to_onehot())
    return BETA*fwd_loss + (1-BETA)*inv_loss


if __name__ == "__main__":
    sun_dataset = SUNDataset(
        path='sun2012/', transforms=[Resize(LENS_SIZE*10), CropToMultiple(LENS_SIZE)])
    data_loader = D.DataLoader(
        dataset=sun_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=BATCH_SIZE,
        collate_fn=SUNDataset.collate
    )

    apnet = ActionPredictor()
    sfpnet = StateFeaturePredictor(apnet.phi)

    optimizer = torch.optim.Adam(nn.ModuleList(
        [apnet, sfpnet]).parameters(), lr=LEARNING_RATE)

    s_t0, a_t0, s_t1 = None, None, None

    ctr = 0

    for batch in data_loader:
        # the environment represents the set of images we're currently training on and knows what region of the image we are at
        env = ActionEnvironment(batch)
        while True:
            s_t1 = env.state  # get current state of the environment

            if ctr > 0:
                a_hat = apnet(s_t0, s_t1)  # inverse module
                phi_hat = sfpnet(s_t0, a_t0)  # forward module

                loss = loss_fn(a_hat, a_t0, phi_hat,
                               apnet.phi(to_phi_input(s_t1)))
                print(ctr, loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ctr += 1
            s_t0 = s_t1
            env.step()
            a_t0 = env.last_action

            if env.done:
                break
