import torch

from models.actor_fc import ActorFC
from models.actor_mha import ActorMHA
from models.architectures.network_structure import NetworkStructure
from models.critic_fc import CriticFC
from models.critic_mha import CriticMHA
from simulation.training_setup import Architecture


def load_actor(architecture: Architecture, network_structure: NetworkStructure, log_std: float, path: str,
               device="cpu") -> ActorFC or ActorMHA:
    if architecture == Architecture.MHA:
        actor = ActorMHA(network_structure.n_features, network_structure.n_layers, network_structure.embed_dim,
                         network_structure.n_heads, network_structure.n_hidden_readout,
                         network_structure.dim_single_pred, log_std, device)
    else:
        actor = ActorFC(network_structure.n_features, network_structure.dim_single_pred, log_std, device)
    data = torch.load(path)
    actor.state = data["Actor"]
    return actor


def load_critic(architecture: Architecture, network_structure: NetworkStructure, log_std: float, path: str,
                device="cpu") -> ActorFC or ActorMHA:
    if architecture == Architecture.MHA:
        critic = CriticMHA(network_structure.n_features, network_structure.n_layers, network_structure.embed_dim,
                           network_structure.n_heads, network_structure.n_hidden_readout,
                           network_structure.dim_single_pred, log_std, device)
    else:
        critic = CriticFC(network_structure.n_features, network_structure.dim_single_pred, log_std, device)
    data = torch.load(path)
    critic.state = data["Critic"]
    return critic
