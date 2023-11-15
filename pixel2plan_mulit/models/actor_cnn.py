import torch as tr
import torch.nn as nn
from models.architectures.cnn import CNN


class Actor(CNN):

    def __init__(self, outdim, n_heads, device="cpu", log_std=0):
        super().__init__()
        self.outdim = outdim
        self.device = device

        self.nn_layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh()
        )

        self.heads = nn.ModuleList([])
        for _ in range(n_heads):
            head = nn.Sequential(nn.Linear(32, self.outdim), )
            self.heads.append(head)

        self.log_std = nn.ParameterList([
            nn.Parameter(tr.ones(1, self.outdim).to(self.device)
                         * log_std, requires_grad=True)
            for _ in range(n_heads)]
        )

    @property
    def n_heads(self):
        return len(self.heads)

    @property
    def params(self):
        params = self.base_params
        for i in range(self.n_heads):
            params.extend(nn.ParameterList(self.heads[i].parameters()).extend([self.log_std[i]]))
        return params

    @property
    def base_params(self):
        cnn_params = nn.ParameterList(self.cnn_layers.parameters())
        return cnn_params.extend(nn.ParameterList(self.nn_layers.parameters()))

    def set_eval(self):
        self.nn_layers.eval()
        self.heads.eval()

    def set_train(self):
        self.nn_layers.train()
        self.heads.train()

    def forward(self, x):
        state, target, idx = x

        f_map = super(Actor, self).forward([state, target])
        f_vec = f_map.view(f_map.size(0), -1)

        a = self.heads[idx](self.nn_layers(f_vec))

        log_std = self.log_std[idx].expand_as(a)
        return a, log_std

    def __call__(self, state, target, index, greedy=False):
        if greedy:
            action, _ = self.forward([state, target, index])
            return action
        else:
            action, log_std = self.forward([state, target, index])
            noise = tr.exp(log_std) * tr.randn_like(action)
            return action + noise.to(self.device)
