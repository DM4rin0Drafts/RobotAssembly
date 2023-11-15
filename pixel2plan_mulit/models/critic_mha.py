from typing import List, Dict

import torch
from torch import Tensor
from models.architectures.multi_head_attention import MPNNMultidimFullAttention


class Critic(MPNNMultidimFullAttention):
    def __init__(self, n_features, n_layers, embed_dim, n_heads, n_hidden_readout, dim_single_pred, control_nodes, device="cpu"):
        super().__init__(n_features, n_layers, embed_dim, n_heads, n_hidden_readout, dim_single_pred, device)
        self.control_nodes = control_nodes
        self.device = device

    def set_train(self):
        self.train()

    def set_eval(self):
        self.eval()

    def forward(self, state_adj: Tensor, actions: Tensor) -> Tensor:
        # TODO: The 4 should ideally be replaced by self.n_features but for the critic n_features is defined differently
        state, adj = state_adj.split([4, state_adj.size()[-2]], dim=-1)

        # Ensure batch_size dimension
        if len(state.size()) < 3:
            state = state.unsqueeze(0)
        if len(actions.size()) < 3:
            actions = actions.unsqueeze(0)

        # TODO: Ensure correct actions for each batch
        batch_size, n_nodes, n_features = state.size()
        dim_single_action = actions.size()[-1]

        state_actions = torch.zeros((batch_size, n_nodes, n_features + dim_single_action), device=self.device)
        state_actions[:, :, :-3] = state.clone().detach()
        state_actions[:, self.control_nodes, -3:] = actions.clone().detach()

        init_node_embeddings = self.initial_embed_layer(state_actions)

        current_node_embeddings = self.mha_layers(init_node_embeddings, adj)

        values = self.readout_layer(*current_node_embeddings)
        return values

    def __call__(self, state_adj: Tensor, actions: Tensor) -> Tensor:
        return self.forward(state_adj, actions)
