from typing import List

import torch
from models.architectures.multi_head_attention import MPNNMultidimFullAttention


class CriticMHA(MPNNMultidimFullAttention):
    def __init__(self, n_features: int, n_layers: int, embed_dim: int, n_heads: int, n_hidden_readout: List,
                 dim_single_pred: int, log_std: float, device="cpu"):
        super().__init__(n_features, n_layers, embed_dim, n_heads, n_hidden_readout, dim_single_pred, device)

        self.n_features = n_features
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_hidden_readout = n_hidden_readout
        self.dim_single_pred = dim_single_pred
        self.log_std = log_std
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

        _state = torch.zeros((batch_size, n_nodes, n_features + dim_single_action), device=self.device)
        _state[:, :, :-3] = state.detach().clone()
        _state[:, -1:, -3:] = action

        init_node_embeddings = self.initial_embed_layer(_state)
        current_node_embeddings = self.mha_layers(init_node_embeddings, adj)
        q_value = self.readout_layer(*current_node_embeddings)
        return q_value.squeeze()

    def __call__(self, state_adj: torch.Tensor, action: torch.Tensor):
        return self.forward(state_adj, action)


def critic_mha_from(critic: CriticMHA) -> CriticMHA:
    """
    Returns a new Critic with equal structure, training setup and device as given one.
    Network parameters are not copied.
    """
    return CriticMHA(critic.n_features, critic.n_layers, critic.embed_dim, critic.n_heads, critic.n_hidden_readout,
                     critic.dim_single_pred, critic.log_std, critic.device)
