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
import skimage.io
import skimage.transform
from enum import Enum
from math import ceil

PREDICT_NEXT_ACTION = False
NUM_ACTIONS = 4 + (1 if PREDICT_NEXT_ACTION else 0)
PHI_SIZE = 32*3*3
LEARNING_RATE = 1e-3
BETA = 0
BATCH_SIZE = 20
SKIP_DECAY_CONSTANT = 0.8**BATCH_SIZE
TEST_SPLIT_PCT = 0.1
LENS_SIZE = 33
IMG_COVG_PCT = 1.5
UPDATE_FREQ = 1000
NUM_EPOCHS = 100

dtypes = torch.cuda if torch.cuda.is_available() else torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_phi_input(s):
    return to_tensor_f(s).view(len(s), 3, LENS_SIZE, -1)


def to_tensor_f(l):
    return to_tensor(l, type=dtypes.FloatTensor)


def to_tensor(l, type=dtypes.LongTensor):
    try:
        return type(l)
    except:
        return l.type(type)


class Action(Enum):
    UP = [-1, 0]
    RIGHT = [0, 1]
    DOWN = [1, 0]
    LEFT = [0, -1]
    NEXT = None

    def to_onehot(self, dtype=torch.long):
        onehot = torch.zeros([BATCH_SIZE, NUM_ACTIONS], dtype=dtype)
        # note that this assumes same action for all training points in minibatch
        for i, a in enumerate(Action):
            if self == list(Action)[i]:
                onehot[:, i] = 1
        return onehot


class CropToMultiple(object):
    def __init__(self, m):
        self.m = m

    def __call__(self, image, fn):
        h, w = image.shape[:2]
        return image[:((h//self.m)*self.m), :((w//self.m)*self.m)]


class Resize(object):
    # Adapted with permission from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, fn):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        try:
            return skimage.transform.resize(image, (new_h, new_w))[:, :, :3]
        except Exception as e:
            print('THE CULPRIT IS ' + fn)


class SUNDataset(D.Dataset):  # TODO
    def __init__(self, path=None, transforms=[], train=True, files=None):
        self.transforms = transforms
        # self.files_accessed = set()
        if files is None:
            self.path = path
            self.files = sorted([os.path.join(self.path, name) for name in os.listdir(
                self.path) if os.path.isfile(os.path.join(self.path, name)) and name.endswith('.jpg')])
            random.shuffle(self.files)
            self.files = self.files[:int(len(self.files) * (1-TEST_SPLIT_PCT))]
            self.other_files = self.files[int(
                len(self.files) * (1-TEST_SPLIT_PCT)):]
            if not train:
                tmp = self.files
                self.files = self.other_files
                self.other_files = tmp
        else:
            self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # self.files_accessed.add(self.files[i])
        img = skimage.io.imread(self.files[i])
        for t in self.transforms:
            img = t(img, self.files[i])
        return img, self.files[i]

    def display_image(self, i):
        plt.figure()
        plt.imshow(self[i])
        plt.show()

    @staticmethod
    def collate(batch):
        return list(zip(*batch))

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
            phi_s_t0 = self.phi(to_phi_input(s_t0))
            a_t0_onehot = a_t0.to_onehot(dtype=torch.float).to(device)[:phi_s_t0.shape[0]]
            v = torch.cat((phi_s_t0, a_t0_onehot), 1)
        return self.fc2(self.fc1(v))


class ActionPolicy():
    def __init__(self, lens_dims):
        self.remaining = to_tensor(
            [int(int(l[0]*l[1])*IMG_COVG_PCT) for l in lens_dims])

    def act(self):
        for r in self.remaining:
            if SKIP_DECAY_CONSTANT**(r.item()/2) >= random.random():
                return Action.NEXT
        self.remaining -= 1
        return random.choice(list(Action)[:-1])
        # return list(Action)[1]


class ActionEnvironment():
    def __init__(self, batch):
        self.batch = batch
        # in units of LENS_SIZE
        self.coords = torch.zeros(
            [len(self.batch), 2], dtype=torch.long).to(device)
        self.dims = to_tensor(
            [[ceil(im.shape[0]/LENS_SIZE), ceil(im.shape[1]/LENS_SIZE)] for im in self.batch])
        self.update_state()
        self.policy = ActionPolicy(self.dims)
        self.done = False
        self.last_action = None
        self.adjusted_actions = torch.zeros(len(self.batch))

    def update_state(self):
        self.state = to_tensor_f([im[int(co[0]*LENS_SIZE):int((co[0]+1)*LENS_SIZE), int(co[1]*LENS_SIZE):int((co[1]+1)*LENS_SIZE)]
                                  for im, co in zip(self.batch, self.coords)])
        # overflow results in non square lens

    def step(self):
        self.last_action = self.policy.act()
        if self.last_action != Action.NEXT:
            self.coords += to_tensor(self.last_action.value)
            # in case of dimension out of bounds, stay at boundary (i.e. take no action)
            overflow_adjustment = to_tensor(self.coords == self.dims)
            underflow_adjustment = to_tensor(self.coords < 0)
            self.adjusted_actions = (
                overflow_adjustment + underflow_adjustment).sum(dim=1) > 0
            self.coords -= overflow_adjustment
            self.coords += underflow_adjustment
            self.update_state()
        else:
            self.done = True


def loss_fn(a_hat, a, phi_hat, phi, adj_acts):
    fwd_loss = F.mse_loss(phi_hat, phi)
    if sum(adj_acts) < len(adj_acts):
        inv_loss = F.cross_entropy(
            a_hat[~adj_acts], a.to_onehot()[:len(adj_acts)].argmax(1)[~adj_acts].to(device))
    else:
        inv_loss = 0
    return BETA*fwd_loss + (1-BETA)*inv_loss


def visualize_env(s_t0, s_t1, a_t0, a_hat):
    fig = plt.figure()
    plt.axis('off')
    rows = 2
    cols = len(a_hat)
    for i in range(1, rows*cols + 1):
        fig.add_subplot(rows, cols, i)
        s_t0_str = 'True action: {}'.format(a_t0)
        s_t1_str = 'Prediction: {}'.format(
            list(Action)[a_hat[(i-1) % len(a_hat)].argmax()])
        plt.text(0, 0, s_t1_str if i > len(a_hat) else s_t0_str)
        plt.imshow((s_t1 if i > len(a_hat) else s_t0)[
                   (i-1) % len(a_hat)], interpolation='nearest')
    plt.show()


def visualize_loss(ctr, loss, acc):
    # adapted with permission from https://matplotlib.org/gallery/api/two_scales.html
    print(ctr, loss, acc)
    plt_xs.append(ctr)
    plt_y1s.append(loss)
    plt_y2s.append(acc)
    ax1.plot(plt_xs, plt_y1s, color='tab:red')
    ax2.plot(plt_xs, plt_y2s, color='tab:blue')
    plt.draw()
    plt.pause(0.001)


def init_viz():
    plt.ion()
    plt.show()
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('acc', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    return ax1, ax2


if __name__ == "__main__":
    sun_dataset = SUNDataset(
        path='sun2012/', transforms=[Resize(LENS_SIZE*5), CropToMultiple(LENS_SIZE)])
    test_sun_dataset = SUNDataset(
        files=sun_dataset.other_files, transforms=sun_dataset.transforms)

    apnet = ActionPredictor().to(device)
    sfpnet = StateFeaturePredictor(apnet.phi).to(device)

    optimizer = torch.optim.Adam(nn.ModuleList(
        [apnet, sfpnet]).parameters(), lr=LEARNING_RATE)

    s_t0, a_t0, s_t1 = None, None, None

    ctr = 0
    batch_ctr = 0
    actions_per_batch = []

    # data viz stuff
    plt_xs = []
    plt_y1s = []
    plt_y2s = []
    ax1, ax2 = init_viz()

    cum_loss = 0
    total_guess = 0
    correct_guess = 0

    data_loader = D.DataLoader(
        dataset=sun_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=BATCH_SIZE,
        collate_fn=SUNDataset.collate
    )

    test_data_loader = D.DataLoader(
        dataset=test_sun_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=BATCH_SIZE,
        collate_fn=SUNDataset.collate
    )

    for i in range(NUM_EPOCHS):
        for idx, batch in enumerate(data_loader):
            # the environment represents the set of images we're currently training on and knows what region of the image we are at
            env = ActionEnvironment(batch[0])
            while True:
                s_t1 = env.state  # get current state of the environment

                if s_t0 is not None:
                    a_hat = apnet(s_t0, s_t1)  # inverse module
                    phi_hat = sfpnet(s_t0, a_t0)  # forward module

                    # manually keep track of action accuracy - 25% is random guess
                    total_guess += len(batch[0])
                    correct_guess += sum(torch.argmax(a_t0.to_onehot(), dim=1).to(device)
                                         == torch.argmax(a_hat, dim=1)).item()

                    # calculate loss
                    loss = loss_fn(a_hat, a_t0, phi_hat,
                                   apnet.phi(to_phi_input(s_t1)), env.adjusted_actions)
                    # print(ctr, loss.item())

                    # visualize loss
                    cum_loss += loss.item()
                    if ctr > 0 and ctr % UPDATE_FREQ == 0:
                        visualize_loss(ctr, cum_loss/ctr,
                                       (correct_guess*100)/total_guess)
                        print('actions per batch: ' +
                              str(sum(actions_per_batch)/len(actions_per_batch)))
                        # visualize_env(s_t0, s_t1, a_t0, a_hat)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                ctr += BATCH_SIZE
                batch_ctr += 1

                s_t0 = s_t1
                env.step()
                a_t0 = env.last_action

                if env.done:
                    if not PREDICT_NEXT_ACTION:
                        s_t0 = None
                        actions_per_batch.append(batch_ctr)
                        batch_ctr = 0
                    break

    cum_loss = 0
    total_guess = 0
    correct_guess = 0

    print('-TESTING----------------------------')

    with torch.no_grad():
        for batch in test_data_loader:
            env = ActionEnvironment(batch[0])
            while True:
                s_t1 = env.state  # get current state of the environment

                if s_t0 is not None:
                    a_hat = apnet(s_t0, s_t1)  # inverse module
                    phi_hat = sfpnet(s_t0, a_t0)  # forward module

                    # manually keep track of action accuracy - 25% is random guess
                    total_guess += len(batch[0])
                    correct_guess += sum(torch.argmax(a_t0.to_onehot()[:a_hat.shape[0]], dim=1)
                                         == torch.argmax(a_hat, dim=1)).item()

                    # calculate loss
                    loss = loss_fn(a_hat, a_t0, phi_hat,
                                   apnet.phi(to_phi_input(s_t1)), env.adjusted_actions)
                    # print(ctr, loss.item())

                    # visualize loss
                    cum_loss += loss.item()
                    if ctr > 0 and ctr % UPDATE_FREQ == 0:
                        visualize_loss(ctr, cum_loss/ctr,
                                       (correct_guess*100)/total_guess)
                        print('actions per batch: ' +
                              str(sum(actions_per_batch)/len(actions_per_batch)))
                        # visualize_env(s_t0, s_t1, a_t0, a_hat)

                ctr += 1
                batch_ctr += 1

                s_t0 = s_t1
                env.step()
                a_t0 = env.last_action

                if env.done:
                    if not PREDICT_NEXT_ACTION:
                        s_t0 = None
                        actions_per_batch.append(batch_ctr)
                        batch_ctr = 0
                    break
