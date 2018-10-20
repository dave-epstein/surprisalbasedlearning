import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path
from collections import OrderedDict
import skimage

NUM_ACTIONS = 5
PHI_SIZE = 32*3*3
LEARNING_RATE = 1e-3
BETA = 0.2
BATCH_SIZE = 3


class SUNDataset(torch.utils.data.Dataset):  # TODO
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        self.filenames = [name for name in os.listdir(
            self.path) if os.path.isfile(os.path.join(self.path, name))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        img = skimage.io.imread(self.filenames[i])
        for t in self.transforms:
            img = t(img)
        return img


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
        phi_xy = torch.cat((self.phi(s_t0), self.phi(s_t1)), 1)
        return self.fc2(self.fc1(phi_xy))


class StateFeaturePredictor(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.fc1 = nn.Linear(PHI_SIZE+NUM_ACTIONS, 256)
        self.fc2 = nn.Linear(256, PHI_SIZE)
        self.phi = phi

    def forward(self, s_t0, a_t0):
        with torch.no_grad():
            # TODO does setting no_grad stop phi from being affected in SFP backprop?
            a_t0_onehot = torch.zeros(BATCH_SIZE, NUM_ACTIONS)
            for i, a in enumerate(a_t0):
                a_t0_onehot[i][a] = 1
            v = torch.cat((self.phi(s_t0), a_t0_onehot), 1)
        return self.fc2(self.fc1(v))


def loss_fn(a_hat, a, phi_hat, phi):
    fwd_loss = F.mse_loss(phi_hat, phi)
    inv_loss = F.cross_entropy(a_hat, a)
    return BETA*fwd_loss + (1-BETA)*inv_loss


apnet = ActionPredictor()
sfpnet = StateFeaturePredictor(apnet.phi)

optimizer = torch.optim.Adam(nn.ModuleList(
    [apnet, sfpnet]).parameters(), lr=LEARNING_RATE)

s_t0, a_t0, s_t1, a_t1 = None, None, None, None

for t in range(500):
    # set last iteration's current state and action to the current iteration's previous state and action
    s_t0 = s_t1
    a_t0 = a_t1

    # for now randomly generate the new state and action
    s_t1 = torch.randn(BATCH_SIZE, 3, 33, 33)
    a_t1 = torch.LongTensor(BATCH_SIZE).random_(NUM_ACTIONS)

    # on the first iteration we don't have a previous, so no predictions
    if t > 0:
        a_hat = apnet(s_t0, s_t1)  # inverse module
        phi_hat = sfpnet(s_t0, a_t0)  # forward module

        loss = loss_fn(a_hat, a_t1, phi_hat, apnet.phi(s_t1))
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
