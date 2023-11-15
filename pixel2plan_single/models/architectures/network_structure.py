from dataclasses import dataclass
from typing import List


@dataclass
class NetworkStructure:
    """
    Defines Multi-Head Attention Neural Network structure. Training is graph-based, features are defined for each node
    in the graph.

    n_features: Number of features per node. E.g., 4 features if each node is represented as
                (x, y, is_target, is_placed). 7 features if each node is represented as (x, y, is_target, is_placed) and
                conditioned on a single action (vx, vy, wz).
    n_layers: Number of multi-head attention layers.
    embed_dim: Dimension of initial (higher dimensional) projection of nodes.
    n_heads: Number of heads per multi-head attention layers.
    n_hidden_readout: Number of hidden layers for readout layer.
    dim_single_pred: Dimension of a single prediction. E.g., a single action has 3 dimensions (vx, vy, wz). Prediction
                     of value of a single state-action pair is a scalar.
    """
    n_features: int
    n_layers: int
    embed_dim: int
    n_heads: int
    n_hidden_readout: List[int]
    dim_single_pred: int
