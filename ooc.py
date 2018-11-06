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
from densenet import DenseNet
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def round_up(n, b):
    t = n % b
    if t < 0:
        t += b
    if t == 0:
        return n
    return n + b - t


parser = argparse.ArgumentParser(
    description='Out of context prediction using surprise-based learning')

parser.add_argument('--batch-size', '-bs', type=int,
                    help='Set the batch size for learning', default=4)
parser.add_argument('--visualize', '-x', type=str2bool,
                    help='Flag for whether matplotlib graphs should be displayed', default=True)
parser.add_argument('--log-freq', '-f', type=int,
                    help='Every how many images we should print an update', default=1000)
parser.add_argument('--parallel', '-p', type=str2bool,
                    help='Flag for whether batch workers should run in parallel', default=True)
parser.add_argument('--num-workers', '-n', type=int,
                    help='How many workers should we use?', default=-1)
parser.add_argument('--test-only', '-t', type=str2bool,
                    help='Run only test?', default=False)
parser.add_argument('--train-only', '-r', type=str2bool,
                    help='Run only train?', default=False)
parser.add_argument('--model-names', '-m',
                    help='Enter the names of the model files to load separated by commas (default is "apnet.pt, sfpnet.pt")', default='apnet.pt,sfpnet.pt')
args = parser.parse_args()

PREDICT_NEXT_ACTION = False
NUM_ACTIONS = 4 + (1 if PREDICT_NEXT_ACTION else 0)
PHI_SIZE = 32*3*3
LEARNING_RATE = 1e-3
BETA = 0.2
BATCH_SIZE = args.batch_size
SKIP_DECAY_CONSTANT = (0.8 if BATCH_SIZE < 64 else 0.9)**BATCH_SIZE
TEST_SPLIT_PCT = 0.1
LENS_SIZE = 64
IMG_COVG_PCT = 1
UPDATE_FREQ = round_up(args.log_freq, BATCH_SIZE)
NUM_EPOCHS = 100
PARALLEL = args.parallel
NUM_WORKERS = args.num_workers if PARALLEL else 0
if NUM_WORKERS < 0 and PARALLEL:
    NUM_WORKERS = args.batch_size
VISUALIZE = args.visualize

dtypes = torch.cuda if torch.cuda.is_available() else torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__" and torch.cuda.is_available():
    try:
        import torch.multiprocessing as multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


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

    def __call__(self, image, file):
        h, w = image.shape[:2]
        return image[:((h//self.m)*self.m), :((w//self.m)*self.m)]


class Resize(object):
    # Adapted with permission from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, file):
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
            print('THE CULPRIT IS ' + file)


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
        return {'img': img, 'file': self.files[i], 'dims': img.shape}

    def display_image(self, i):
        plt.figure()
        plt.imshow(self[i])
        plt.show()

    @staticmethod
    def collate(batch):
        max_dims = to_tensor([_['img'].shape for _ in batch],
                             type=torch.LongTensor).max(dim=0)[0]
        for el in batch:
            pad = (max_dims - to_tensor(el['dims'],
                                        type=torch.LongTensor))[:-1]
            el['img'] = F.pad(to_tensor(el['img'], type=torch.FloatTensor),
                              (0, 0, 0, pad[1].item(), 0, pad[0].item()))
        return D.dataloader.default_collate(batch)

    # def done(self):
    #     return len(self.files_accessed) == len(self)


class FeatureMapper(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = nn.Sequential(OrderedDict(
        #     [
        #         ('conv1', nn.Conv2d(3, 32, 3, stride=2, padding=1)),
        #         ('elu1', nn.ELU()),
        #         ('conv2', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
        #         ('elu2', nn.ELU()),
        #         ('conv3', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
        #         ('elu3', nn.ELU()),
        #         ('conv4', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
        #         ('elu4', nn.ELU()),
        #     ]
        # ))
        self.net = DenseNet(growth_rate=12, drop_rate=0.2,
                            block_config=(4, 4))
        self.out_features = self.net.num_features

    def forward(self, x):
        # return self.flatten(self.net(x))
        return self.net(x)

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
        # self.fc1 = nn.Linear(2*PHI_SIZE, 256)
        self.fc1 = nn.Linear(self.phi.out_features*2, 256)
        self.fc2 = nn.Linear(256, NUM_ACTIONS)

    def forward(self, s_t0, s_t1):
        phi_cat = torch.cat((self.phi(to_phi_input(s_t0)),
                             self.phi(to_phi_input(s_t1))), 1)  # concat the two state vectors as input to the mlp
        return self.fc2(self.fc1(phi_cat))


class StateFeaturePredictor(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.phi = phi
        self.fc1 = nn.Linear(self.phi.out_features+NUM_ACTIONS, 256)
        self.fc2 = nn.Linear(256, self.phi.out_features)

    def forward(self, s_t0, a_t0):
        with torch.no_grad():
            # TODO does setting no_grad stop phi from being affected in SFP backprop?
            phi_s_t0 = self.phi(to_phi_input(s_t0))
            a_t0_onehot = a_t0.to_onehot(dtype=torch.float).to(device)[
                :phi_s_t0.shape[0]]
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
        self.state = torch.stack([im[int(co[0]*LENS_SIZE):int((co[0]+1)*LENS_SIZE), int(co[1]*LENS_SIZE):int((co[1]+1)*LENS_SIZE)]
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
            self.policy.remaining += to_tensor(self.adjusted_actions)
            self.coords -= overflow_adjustment
            self.coords += underflow_adjustment
            self.update_state()
        else:
            self.done = True


def loss_fn(a_hat, a, phi_hat, phi, adj_acts):
    fwd_loss = F.mse_loss(phi_hat, phi)
    if sum(to_tensor(adj_acts)) < len(adj_acts):
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


def visualize_loss(ctr, loss, acc, viz):
    # adapted with permission from https://matplotlib.org/gallery/api/two_scales.html
    print(ctr, loss, acc)
    if viz:
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


def preprocess_batch(batch):
    batch['dims'] = torch.stack(batch['dims']).t()
    batch['img'] = [_[[slice(__.item()) for __ in batch['dims'][i]]]
                    for i, _ in enumerate(batch['img'])]
    return batch


if __name__ == "__main__":
    sun_dataset = SUNDataset(
        path='sun2012/', transforms=[Resize(LENS_SIZE*7), CropToMultiple(LENS_SIZE)])
    test_sun_dataset = SUNDataset(
        files=sun_dataset.other_files, transforms=sun_dataset.transforms)

    apnet = ActionPredictor().to(device)
    sfpnet = StateFeaturePredictor(apnet.phi).to(device)

    if not args.test_only:
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

        if VISUALIZE:
            ax1, ax2 = init_viz()

        cum_loss = 0
        total_guess = 0
        correct_guess = 0

        data_loader = D.DataLoader(
            dataset=sun_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn=SUNDataset.collate
        )

        test_data_loader = D.DataLoader(
            dataset=test_sun_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn=SUNDataset.collate
        )

        best_acc = 0

        for i in range(NUM_EPOCHS):
            print('starting epoch', i)
            for idx, batch in enumerate(data_loader):
                batch = preprocess_batch(batch)
                # the environment represents the set of images we're currently training on and knows what region of the image we are at
                env = ActionEnvironment(batch['img'])
                while True:
                    s_t1 = env.state  # get current state of the environment

                    if s_t0 is not None:
                        a_hat = apnet(s_t0, s_t1)  # inverse module
                        phi_hat = sfpnet(s_t0, a_t0)  # forward module

                        # manually keep track of action accuracy - 25% is random guess
                        total_guess += len(batch['img'])
                        correct_guess += sum(torch.argmax(a_t0.to_onehot()[:a_hat.shape[0]], dim=1).to(device)
                                             == torch.argmax(a_hat, dim=1)).item()

                        # calculate loss
                        loss = loss_fn(a_hat, a_t0, phi_hat,
                                       apnet.phi(to_phi_input(s_t1)), env.adjusted_actions)
                        # print(ctr, loss.item())

                        # visualize loss
                        cum_loss += loss.item()
                        if ctr > 0 and UPDATE_FREQ > 0 and ctr % UPDATE_FREQ == 0:
                            visualize_loss(ctr, cum_loss/ctr,
                                           (correct_guess*100)/total_guess, VISUALIZE)
                            if len(actions_per_batch) > 0:
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

            if correct_guess/total_guess > best_acc:
                best_acc = correct_guess/total_guess
                print('new best accuracy', i, (correct_guess*100)/total_guess)

            torch.save(apnet.state_dict(), 'apnet{0}-acc{1}.pt'.format(i,
                                                                       round((correct_guess*100)/total_guess, 2)))
            torch.save(sfpnet.state_dict(), 'sfpnet{0}.pt'.format(i))

            cum_loss = 0
            total_guess = 0
            correct_guess = 0

        # torch.save(apnet, 'apnet-final.pt')
    else:
        model_names = args.model_names.split(',')
        apnet.load_state_dict(torch.load(model_names[0]))
        sfpnet.load_state_dict(torch.load(model_names[1]))

    if not args.train_only:
        with torch.no_grad():
            print('TESTING ACTION RECOGNITION ACCURACY')
            for batch in test_data_loader:
                batch = preprocess_batch(batch)
                env = ActionEnvironment(batch['img'])
                while True:
                    s_t1 = env.state  # get current state of the environment

                    if s_t0 is not None:
                        a_hat = apnet(s_t0, s_t1)  # inverse module
                        # phi_hat = sfpnet(s_t0, a_t0)  # forward module

                        # manually keep track of action accuracy - 25% is random guess
                        total_guess += len(batch['img'])
                        correct_guess += sum(torch.argmax(a_t0.to_onehot()[:a_hat.shape[0]], dim=1)
                                             == torch.argmax(a_hat, dim=1)).item()

                        # calculate loss
                        loss = loss_fn(a_hat, a_t0, phi_hat,
                                       apnet.phi(to_phi_input(s_t1)), env.adjusted_actions)
                        # print(ctr, loss.item())

                        # visualize loss
                        cum_loss += loss.item()
                        if ctr > 0 and ctr % UPDATE_FREQ == 0 and UPDATE_FREQ > 0:
                            visualize_loss(ctr, cum_loss/ctr,
                                           (correct_guess*100)/total_guess, VISUALIZE)
                            if len(actions_per_batch) > 0:
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
