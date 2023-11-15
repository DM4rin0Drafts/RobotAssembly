import torch
from utilities.utilities import random_noise
from models.architectures.fully_connected import FullyConnected


class ActorFC(FullyConnected):
    def __init__(self, n_features: int, dim_single_pred: int, log_std: float, small_fc: bool, device="cpu"):
        super().__init__(in_features=2 * (4 + 4 + 1), out_features=dim_single_pred, small=small_fc, device=device)

        self.n_features = n_features
        self.dim_single_pred = dim_single_pred
        self.log_std = log_std
        self.small_fc = small_fc
        self.device = device

    def forward(self, state_adj: torch.Tensor):
        state, adj = state_adj.split([self.n_features, state_adj.size()[-2]], dim=-1)

        # Ensure batch_size dimension
        if len(state.size()) < 3:
            state = state.unsqueeze(0)

        batch_size, n_nodes, n_features = state.size()

        state = state[..., :2]  # only use the coordinates of the states and throw away the bool flags
        state = state.reshape(batch_size, -1)  # flatten, but keep batch dimensions

        actions = self.layers(state)
        return actions.squeeze()

    def __call__(self, state_adj: torch.Tensor, greedy=False):
        tanh = torch.nn.Tanh()
        if greedy:
            return tanh(self.forward(state_adj))
        else:
            action = self.forward(state_adj)
            noise = random_noise(self.log_std, action, self.device).to(self.device)
            return tanh(action + 0.1 * noise)


def actor_fc_from(actor: ActorFC) -> ActorFC:
    """
    Returns a new Actor with equal structure, training setup and device as given one.
    Network parameters are not copied.
    """
    return ActorFC(actor.n_features, actor.dim_single_pred, actor.log_std, actor.small_fc, actor.device)
