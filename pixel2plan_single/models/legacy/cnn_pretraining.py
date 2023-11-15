from tqdm import tqdm
from collections import deque
import torch as tr
import numpy as np
import random
from environment.token import Token
from environment.tangram import Tangram
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt


class ReplayBuffer(object):
    def __init__(self, buffer_size, device="cpu"):
        self.count = 0
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.device = device

    def add(self, samples):
        for sample in samples:
            if self.count < self.buffer_size:
                self.buffer.append(sample)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(sample)

    def sample(self, batch_size):
        """ Sample batch of batch_size. If less samples are stored return all. """
        # Check current number of stored transitions
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch, t_batch = list(map(tr.stack, list(zip(*batch))))

        return s_batch.to(self.device), t_batch.to(self.device)

    def clear(self):
        self.buffer.clear()
        self.count = 0


class CNN(nn.Module):
    """Defines CNN to feature generation"""

    def __init__(self, device):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ).to(device=device)

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #
            nn.BatchNorm2d(2),
            nn.ConvTranspose2d(2, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(256),
            nn.Sigmoid(),
        ).to(device=device)

    @property
    def encoder_state(self):
        state_dict = self.encoder.state_dict().copy()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        return state_dict

    @property
    def state(self):
        state_dict = self.state_dict().copy()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        return state_dict

    @state.setter
    def state(self, state_dict):
        self.load_state_dict(state_dict)

    def forward(self, x):
        """Function returning the neural network output for a given input x. """
        state, target = x

        array = target.float() - state.float()

        encoded = self.encoder(array)
        decoded = self.decoder(encoded)

        return decoded


class Trajectory(object):

    def __init__(self):
        self.transitions = None

    @property
    def length(self):
        if self.transitions:
            return len(self.transitions)
        else:
            return 0

    def add_transition(self, transition):
        if self.length > 0:
            self.transitions[-1].next_transition = transition
            self.transitions.append(transition)
        else:
            self.transitions = [transition]

    @property
    def data(self):
        """ Provides all transitions in order of their execution """
        return [t.data for t in self.transitions]


class Transition(object):
    def __init__(self, state, next_state):
        self.state = state
        self.next_state = next_state
        self.__next_transition = None

    @property
    def target(self):
        if self.__next_transition:
            return self.__next_transition.target
        else:
            return self.next_state

    @property
    def terminal(self):
        if self.next_transition:
            return tr.tensor(0, dtype=tr.float32).to(self.state.device)
        else:
            return tr.tensor(1, dtype=tr.float32).to(self.state.device)

    @property
    def next_transition(self):
        return self.__next_transition

    @next_transition.setter
    def next_transition(self, value):
        # assert self.next_state == value.state, "Next transition does not match."
        self.__next_transition = value

    @property
    def data(self):
        return [self.state, self.target]


def create_target():
    game.reset_tokens()
    token_order = np.random.permutation(game.n_token)
    x, y = np.clip(np.random.normal(0.0, 0.5, 2), -1, 1).astype(np.float32)

    samples = Trajectory()
    for idx in token_order:
        state = game.state.squeeze(0)
        phi = np.random.uniform(-1, 1, 1).astype(np.float32)
        game.set_token_position_to(idx, x, y, phi)
        samples.add_transition(Transition(state, game.state.squeeze(0)))
    return samples.data


def fill_buffer(buffer, n=None):
    if not n:
        n = buffer.buffer_size
    for _ in tqdm(range(int(n/2))):
        buffer.add(create_target())
    return


def main(episodes, offset=0):
    criterion = nn.BCELoss()

    for i in tqdm(range(episodes)):
        optimizer.zero_grad()
        samples = buffer.sample(256)
        output = network.forward(samples)
        loss = criterion(output, samples[1].float() - samples[0].float())
        loss.backward()
        optimizer.step()
        writer.add_scalar('AutoEncoder_Loss', loss.item(), offset+i)


def plot(input, reconstruction):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(input, 'gray')
    ax2.imshow(reconstruction, 'gray')
    ax1.set_title("Input")
    ax2.set_title("Reconstruction")
    return fig


def test_reconstruction(episode=0):
    n_tests = 10
    testbuffer = ReplayBuffer(n_tests, device='cuda:0')
    fill_buffer(testbuffer, n_tests)

    samples = testbuffer.sample(n_tests)
    output = network.forward(samples).detach()
    for i in range(n_tests):
        input = (samples[1][i].float() - samples[0][i].float()).squeeze(0).cpu()
        fig = plot(input, output[i].cpu().squeeze(0))
        writer.add_figure(f'Reconstruction Test:', fig, episode)
        # plt.show()


def save_encoder():
    _run = writer.get_logdir()

    _path = os.path.join(os.getcwd(), f'{_run}/data.tar')
    print(f'Saving Encoder in run: {_run} ...')
    _dict = {
        'Encoder': network.encoder_state,
    }
    tr.save(_dict, _path)
    print('Data saved...')


def save_progress():
    _run = writer.get_logdir()

    _path = os.path.join(os.getcwd(), f'{_run}/data.tar')
    print(f'Saving data in run: {_run} ...')
    _dict = {
        'CNN': network.state,
        'Buffer':  buffer,
        'Optimizer': optimizer.state_dict(),
    }
    tr.save(_dict, _path)
    print('Data saved...')


def load(_run):
    _path = os.path.join(os.getcwd(), f'runs/{_run}/data.tar')
    if os.path.isfile(_path):
        _dict = tr.load(_path)
        network.state = _dict['CNN']
        buffer = _dict['Buffer']
        optimizer.load_state_dict(_dict['Optimizer'])
        return buffer
    else:
        raise OSError('file not found')


if __name__ == '__main__':
    t = [Token(name="triangle2"), Token(name="square")]
    game = Tangram(tokenlist=t, gui=False, device='cpu')
    network = CNN('cuda:0')

    writer = SummaryWriter()
    optimizer = optim.Adam(network.parameters(), lr=1e-3)

    buffer = ReplayBuffer(100000, device='cuda:0')
    fill_buffer(buffer)
    # buffer = load('Enter Name Here')  # Ensure the run contains a data.tar

    main(50000, offset=0)

    network.eval()
    test_reconstruction()

    save_progress()
    # save_encoder()

    print("Done.")
