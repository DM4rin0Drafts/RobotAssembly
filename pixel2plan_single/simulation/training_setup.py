from dataclasses import dataclass
from enum import Enum, auto


class Reward(Enum):
    DENSE = auto()
    SPARSE = auto()


class Architecture(Enum):
    MHA = auto()
    FC_Small = auto()
    FC_Large = auto()


@dataclass
class TrainingSetup:
    """
    n_episodes: Number of episodes executed by the master.
    max_pending_sims: Maximum number of pending rollout requests.
    n_iterations: Number of rollouts per request/episode.
    batch_size: Number of samples used for updates.
    buffer_size: Size of the replay buffer.
    lr_actor: Learning rate actor network.
    lr_critic: Learning rate critic network.
    log_std: Logarithm of standard deviation used for Gaussian random noise to enable exploration (non-greedy forward
                calls) for Actor and Critic networks.
    gamma: Discount factor for reward.
    tau: Interpolation factor for polyak averaging for target networks.
    reward: Specifies reward function to use.
    architecture: Specifies architecture to use.
    overfit: Whether to use single (hardcoded) spawn and target position to test if the network is able to overfit.
    """
    n_episodes: int
    max_pending_sims: int
    n_iterations: int

    batch_size: int
    buffer_size: int

    lr_actor: float
    lr_critic: float
    log_std: float
    gamma: float
    tau: float
    reward: Reward
    compare_rewards: bool

    architecture: Architecture

    overfit: bool
