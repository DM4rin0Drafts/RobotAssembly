import argparse
import os


def settings(parser):
    # General
    parser.add_argument('--n_episodes', type=int, default=100000, help="Number of episodes executed by the master.")
    parser.add_argument('--max_pending_sims', type=int, default=6, help="Maximum number of pending rollout requests.")
    parser.add_argument('--n_iterations', type=int, default=32, help="Number of rollouts per request/episode.")
    parser.add_argument('--n_tests', type=int, default=10, help="Number of tests performed in the demo.")
    parser.add_argument('--seed', type=int, default=0, help="Seed for random numbers.")

    # Optimization
    parser.add_argument('--lr_actor', type=float, default=5e-4, help="Learning rate for actor networks.")
    parser.add_argument('--lr_critic', type=float, default=1e-3, help="Learning rate for critic networks.")
    parser.add_argument('--log_std', type=float, default=0.,
                        help="Logarithm of standard deviation used for Gaussian random noise to enable exploration"
                             "(non-greedy forward calls) for Actor and Critic networks.")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for reward calculation.")
    parser.add_argument('--tau', type=float, default=1e-3,
                        help="Interpolation factor for polyak averaging for target networks.")
    parser.add_argument('--batch_size', type=int, default=64, help="Number of samples used for updates.")
    parser.add_argument('--buffer_size', type=int, default=12500, help="Size of the replay buffer.")
    parser.add_argument('--dense', action="store_true", default=False,
                        help="Flag for using dense reward (center distance) during training.")
    parser.add_argument('--compare_rewards', action="store_true", default=False,
                        help="Flag for comparing sparse and dense rewards in tests "
                             "(independent of reward used for training)")

    # Network Architecture
    parser.add_argument('--architecture', type=str, default="MHA",
                        help="Architecture to use for Actor and Critic. Options are \"MHA\" for Multihead-Attention, "
                             "\"FC_Small\" for a Fully Connected network with a reduced number of parameters , "
                             "\"FC_Large\" for a Fully Connected network with similar number of parameters as "
                             "MHA. Per default Multihead-Attention architecture is used.")

    # Environment
    parser.add_argument('--gui', action="store_true", default=False, help="Flag for activating bullet gui.")
    parser.add_argument('--tokens', nargs="*", type=list, default=["square"])
    parser.add_argument('--debug', action="store_true", default=False, help="Flag for activating PyCharm debugging.")
    parser.add_argument('--overfit', action="store_true", default=False,
                        help="If set, use a single spawn and target position to test whether the network is able to "
                             "overfit.")


def parse_args(*args, **kwargs):
    # create ArgumentParser object
    parser = argparse.ArgumentParser('Tangram Game')

    # optional arguments
    settings(parser)

    # optional argument load data
    parser.add_argument(f'-l', f'--load', type=file_loader, nargs='?', const=file_loader(''), default=False,
                        help=f'Load data from /runs/. Just -l loads data from latest run. '
                             f'-l X loads data from run X.')

    parser.add_argument(f'-t', f'--load_target', type=target_loader, nargs='?', const=target_loader(''), default=False,
                        help=f'Load target image from /. Just -t loads data from latest run. '
                             f'-t X loads target image from path X.')

    return parser.parse_args(*args, **kwargs)


def file_loader(_str):
    _runs = sorted(os.listdir(os.path.join(os.getcwd(), f'runs/')))
    if not _runs:
        return False
    elif _str in _runs:
        return _str
    else:
        return _runs[-1]


def target_loader(_str):
    target_image = os.path.join(os.getcwd(), _str)
    assert os.path.exists(target_image)
    return target_image
