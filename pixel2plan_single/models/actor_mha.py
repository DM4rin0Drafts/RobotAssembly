from typing import List

import torch
from models.architectures.multi_head_attention import MPNNMultidimFullAttention
from utilities.utilities import random_noise


class ActorMHA(MPNNMultidimFullAttention):
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

    def forward(self, state_adj: torch.Tensor):
        state, adj = state_adj.split([self.n_features, state_adj.size()[-2]], dim=-1)

        # Ensure batch_size dimension
        if len(state.size()) < 3:
            state = state.unsqueeze(0)

        init_node_embeddings = self.initial_embed_layer(state)
        current_node_embeddings = self.mha_layers(init_node_embeddings, adj)
        actions = self.readout_layer(*current_node_embeddings)
        return actions[:, -1]

    def __call__(self, state_adj: torch.Tensor, greedy=False):
        tanh = torch.nn.Tanh()
        if greedy:
            return tanh(self.forward(state_adj))
        else:
            action = self.forward(state_adj)
            noise = random_noise(self.log_std, action, self.device).to(self.device)
            return tanh(action + 0.1 * noise)


def actor_mha_from(actor: ActorMHA) -> ActorMHA:
    """
    Returns a new Actor with equal structure, training setup and device as given one.
    Network parameters are not copied.
    """
    return ActorMHA(actor.n_features, actor.n_layers, actor.embed_dim, actor.n_heads, actor.n_hidden_readout,
                    actor.dim_single_pred, actor.log_std, actor.device)
