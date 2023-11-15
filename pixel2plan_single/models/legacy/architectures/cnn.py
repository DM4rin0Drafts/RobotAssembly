import torch.nn as nn


class CNN(nn.Module):
    """Defines CNN to feature generation"""

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            #
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=5, stride=5),
        )

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
        return self.cnn_layers(array).flatten(start_dim=1)
