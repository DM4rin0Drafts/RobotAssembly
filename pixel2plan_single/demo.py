import numpy as np
import torch as tr
import random
import matplotlib.pyplot as plt

from environment.token import Token
from utilities.argumentparser import *
from simulation.master import Master

""" Demo for trained Tangram policies
Please specify the correct token in settings in correct order to match the training case of a given policy.
To run use: python -u demo.py --load=<run name>
Here <run name> is relative to the pixel2plan/runs/ folder. If the specified run does not exist, the run at the bottom
of the runs folder is used as default. 
To see results use: Tensorboard --logdir=runs --samples_per_plugin=images=<n_tests>
Default for <n_tests> is set to 100. Without samples_per_plugin Tensorboard can only display 10 images.
"""

tr.manual_seed(0)
np.random.seed(0)
random.seed(0)


def demo():
    args = parse_args()

    print("\n\nInitializing Demo for Policy: ", args.load)

    args.load_target = True

    master = Master(args, device="cpu", demo=True)


    print("Initializing Target image: ", args.load_target, "\n\n")
    if args.load_target:
        master.load_target("environment/targets/square1_400x400.png")
        args.n_tests = 1

    print(f"\nCommencing Test")
    master.test(0, n_tests=args.n_tests)

    print("\nSuccessfully finished Demo.\n")


if __name__ == '__main__':
    demo()