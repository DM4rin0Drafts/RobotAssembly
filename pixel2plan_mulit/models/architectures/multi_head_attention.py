"""
Implementation based on:
* https://github.com/nifunk/GNNMushroomRL
* Learn2Assemble with Structured Representations and Search for Robotic Architectural Construction, Niklas Funk,
    Georgia Chalvatzaki, Boris Belousov, Jan Peters. https://proceedings.mlr.press/v164/funk22a/funk22a.pdf
* Attention is All you Need, Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
    Łukasz Kaiser, Illia Polosukhin. https://papers.nips.cc/paper/7181-attention-is-all-you-%0Aneed.pdf
"""
# TODO: Maybe we can make the condition on action specific to forward call. Problem are the readout layer dimensions.
# TODO: Extend to allow for multiple tokens

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


class SkipConnection(nn.Module):
    """ Skip connection that allows for forward passes that return the result and the mask. """

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input0, input1):
        res1, res2 = self.module(input0, input1)
        return (input0 + res1), input1


class SkipConnectionSimple(nn.Module):
    """ Skip connection that does not allow for forward passes that also return the mask. """

    def __init__(self, module):
        super(SkipConnectionSimple, self).__init__()
        self.module = module

    def forward(self, input0, input1):
        res = self.module(input0)
        return (input0 + res), input1


class CustomSequential(nn.Sequential):
    """ Custom definition of a sequential layer that allows to pass multiple arguments in the forward call. """

    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class ReadoutLayerMultidim(nn.Module):
    def __init__(self, embed_dim, n_layers_hidden=[], output_dim=1, bias_pool=False, bias_readout=True, device="cpu"):
        """
        Args:
            embed_dim: Dimension of encoded graph.
            n_layers_hidden: Number of hidden layers.
            output_dim: Dimension of output.
            bias_pool: Indicates whether to add bias to pooled layer.
            bias_readout: Indicates whether to add bias to readout layers.
        """
        super().__init__()

        self._is_critic = output_dim == 1

        self.layer_pooled = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias_pool, device=device)

        if type(n_layers_hidden) != list:
            n_layers_hidden = [n_layers_hidden]
        # TODO: At this point 'pretty' conditioning becomes tricky
        first_input_dim = [embed_dim] if self._is_critic else [2 * embed_dim]
        n_layers_hidden = first_input_dim + n_layers_hidden + [output_dim]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_layers_hidden, n_layers_hidden[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout, device=device)
            self.layers_readout.append(layer)
        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings, adj):
        # Global features: Normalized sum over all node embeddings
        h_pooled = self.layer_pooled(node_embeddings.sum(dim=1) / node_embeddings.shape[1])

        if self._is_critic:
            features = F.relu(h_pooled)
        else:
            f_local = node_embeddings

            # Scale up global features to number of nodes
            f_pooled = h_pooled.repeat(1, f_local.shape[1]).view(f_local.shape)

            # Combine global and local features
            features = F.relu(torch.cat([f_pooled, f_local], dim=-1))

        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i < len(self.layers_readout) - 1:
                features = F.relu(features)

        return features


# TODO: Why don't we use torch.nn.MultiheadAttention()?
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None, device="cpu"):
        """
        Args:
            input_dim: Dimension of inputs.
            embed_dim: Dimension of embedding layer output.
            n_heads: Number of attention layers.
            val_dim: Dimension of values.
            key_dim: Dimension of keys. Same dimension applies to queries.
        """
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / np.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim).to(device))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim).to(device))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim).to(device))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim).to(device))

        self.init_parameters()

    def init_parameters(self):
        """ Initialize parameters by Xavier initialization. """
        for param in self.parameters():
            stdv = 1 / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, mask, h=None):
        """
        Args:
             q (batch_size, n_query, input_dim): Queries
             h (batch_size, graph_size, input_dim): Data
             mask (batch_size, n_query, graph_size): Mask with 1 if attention not possible (negative adjacency)
        """
        if h is None:
            h = q  # self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input."

        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = q.contiguous().view(-1, input_dim)

        shape_q = (self.n_heads, batch_size, n_query, -1)
        shape_k_v = (self.n_heads, batch_size, graph_size, -1)

        Q = torch.matmul(q_flat, self.W_query).view(shape_q)
        K = torch.matmul(h_flat, self.W_key).view(shape_k_v)
        V = torch.matmul(h_flat, self.W_val).view(shape_k_v)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[torch.logical_not(mask.bool())] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # Set attention to 0 for nodes without neighbors
        if mask is not None:
            attn.clone()[torch.logical_not(mask.bool())] = 0

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out, mask


class MultiHeadFullAttentionLayer(CustomSequential):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_hidden_feed_forward=512,
            device="cpu"):
        """
        Args:
            n_heads: Number of heads.
            embed_dim: Dimension of embedded graph.
            n_hidden_feed_forward: Dimension of feed forward layer.
        """
        super(MultiHeadFullAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(n_heads=n_heads, input_dim=embed_dim, embed_dim=embed_dim, device=device)
            ),
            SkipConnectionSimple(
                nn.Sequential(
                    nn.Linear(in_features=embed_dim, out_features=n_hidden_feed_forward, device=device),
                    nn.ReLU(),
                    nn.Linear(in_features=n_hidden_feed_forward, out_features=embed_dim, device=device)
                ) if n_hidden_feed_forward > 0 else nn.Linear(embed_dim, embed_dim)
            ),
        )


class MPNNMultidimFullAttention(nn.Module):
    def __init__(self,
                 n_features,
                 n_layers=3,
                 embed_dim=64,
                 n_heads=4,
                 n_hidden_readout=[],
                 dim_single_pred=None,
                 device="cpu"):
        """
        Args:
            n_features: Number of features per node
            embed_dim: Dimension of initial (higher dimensional) projection of nodes.
            n_layers: Number of multi-head attention layers.
            n_heads: Number of heads for multi-head attention layers.
            n_hidden_readout: Number of hidden layers for readout layer.
            dim_single_pred: Dimension of single prediction. E.g., single action has dim 3 (vx, vy, wz).
        """
        super().__init__()

        self.n_features = n_features

        self.initial_embed_layer = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=embed_dim, bias=False, device=device),
            nn.ReLU()
        )
        self.initial_embed_layer.apply(self.init_weights)  # initialize the parameters of Linear and ReLU layer

        feed_forward_hidden = 512
        self.mha_layers = CustomSequential(
            *(MultiHeadFullAttentionLayer(n_heads, embed_dim, feed_forward_hidden, device) for _ in range(n_layers))
        )

        self.readout_layer = ReadoutLayerMultidim(embed_dim, n_hidden_readout, dim_single_pred, device=device)

    @property
    def state(self):
        state_dict = self.state_dict().copy()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        return state_dict

    @state.setter
    def state(self, state_dict):
        self.load_state_dict(state_dict)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain("relu"))
