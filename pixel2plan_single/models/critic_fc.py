import torch
from models.architectures.fully_connected import FullyConnected


class CriticFC(FullyConnected):
    def __init__(self, n_features: int, dim_single_pred: int, log_std: float, small_fc: bool, device="cpu"):
        super().__init__(in_features=2 * (4 + 4 + 1) + 3, out_features=dim_single_pred, small=small_fc, device=device)

        self.n_features = n_features
        self.dim_single_pred = dim_single_pred
        self.log_std = log_std
        self.small_fc = small_fc
        self.device = device

    def forward(self, state_adj: torch.Tensor, action: torch.Tensor = None):
        # TODO: The 4 should ideally be replaced by self.n_features but for the critic n_features is defined differently
        state, adj = state_adj.split([4, state_adj.size()[-2]], dim=-1)

        # Ensure batch_size dimension
        if len(state.size()) < 3:
            state = state.unsqueeze(0)
        if len(action.size()) < 3:
            action = action.unsqueeze(1)

        batch_size, n_nodes, n_features = state.size()
        dim_single_action = action.size()[-1]

        state = state[..., :2]  # only use the coordinates of the states and throw away the bool flags
        state = state.reshape(batch_size, -1)  # flatten, but keep batch dimensions
        action = action.reshape(batch_size, dim_single_action)  # flatten, but keep batch dimensions
        state = torch.cat((state, action), dim=-1)

        q_value = self.layers(state)

        return q_value.squeeze()

    def __call__(self, state_adj: torch.Tensor, action: torch.Tensor):
        return self.forward(state_adj, action)


def critic_fc_from(critic: CriticFC) -> CriticFC:
    """
    Returns a new Critic with equal structure, training setup and device as given one.
    Network parameters are not copied.
    """
    return CriticFC(critic.n_features, critic.dim_single_pred, critic.log_std, critic.small_fc, critic.device)
