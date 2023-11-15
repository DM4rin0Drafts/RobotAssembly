import logging
import os
import time
from typing import List, Tuple

import numpy
import numpy as np
from matplotlib import pyplot as plt

from constants import PATH_PROJECT, actor_structure, critic_structure
from environment import Tangram, Token
from models.actor_fc import ActorFC
from models.actor_mha import ActorMHA
from models.critic_fc import CriticFC
from models.critic_mha import CriticMHA
from simulation.training_setup import Architecture
from utilities.models import load_actor, load_critic


class Visualizer:
    def __init__(self, token: str, actor: ActorFC or ActorMHA, critic: CriticFC or CriticMHA):
        self.env = Tangram([Token(token)], gui=True)

        self.actor = actor
        self.critic = critic

    def actor_single_step(self, center_target: Tuple[float, float], orientation_target: float,
                          center_state: Tuple[float, float], orientation_state: float, create_plots=True,
                          sleep_before=1, sleep_after=1000):
        print("Creating setup:\n\tTarget:\t{}\n\tState:\t{}".format([center_target, orientation_target],
                                                                    [center_state, orientation_state]))

        self.env.create_constant_setup(center_target, orientation_target, center_state, orientation_state)
        target_img = self.env.target_img.squeeze()
        state_img = self.env.state_img.squeeze()

        state_target = self.env.target
        self.connect_corners(state_target[:, :2])

        time.sleep(sleep_before)

        action = self.actor(self.env.state, greedy=True)[0]
        vx, vy, wz = action.detach().numpy()
        self.env.tokens[0].move(vx, vy, wz)
        pred_img = self.env.state_img.squeeze()
        print("Predicted action:\t{}\n".format([vx, vy, wz]))

        if create_plots:
            fig, axs = plt.subplots(1, 4)
            axs[0].imshow(state_img, cmap="gray")
            axs[0].set_title("Initial")
            axs[0].axis("off")

            axs[1].imshow(target_img, cmap="gray")
            axs[1].set_title("Target")
            axs[1].axis("off")

            axs[2].imshow(pred_img, cmap="gray")
            axs[2].set_title("Predicted")
            axs[2].axis("off")

            axs[3].imshow((target_img + pred_img) / 2, cmap="gray")
            axs[3].set_title("Overlap")
            axs[3].axis("off")
            plt.show()

        time.sleep(sleep_after)

    def actor_multi_step_rand(self, n_steps: int, sleep=1):
        for _ in range(n_steps):
            target, state = np.random.uniform(-1, 1, [2, 3])
            self.actor_single_step(target[:2], target[-1], state[:2], state[-1], sleep_before=sleep, sleep_after=sleep)
            self.env.client.removeAllUserDebugItems()
        return

    def connect_corners(self, corners: numpy.ndarray, line_color=[1, 0, 0], line_width=2):
        _corners = [list(c) for c in corners]
        pairs = [(_corners[c1], _corners[c2]) for c1 in range(len(_corners)) for c2 in range(c1 + 1, len(_corners))]

        client = self.env.client
        for p in pairs:
            p1 = p[0]
            p2 = p[1]
            client.addUserDebugLine([p1[0], p1[1], 0], [p2[0], p2[1], 0], lineColorRGB=line_color,
                                    lineWidth=line_width)


if __name__ == "__main__":
    path = os.path.join(PATH_PROJECT, "runs", "Sep07_20-02-09_gaqc0002_MHA_SPARSE", "data0.tar")
    architecture = Architecture.MHA
    log_std = 0.

    actor = load_actor(architecture, actor_structure, log_std, path)
    critic = load_critic(architecture, critic_structure, log_std, path)

    visualizer = Visualizer("square", actor, critic)

    """visualizer.actor_single_step(center_target=(-0.5, 0.), orientation_target=0., center_state=(0., 0.),
                                 orientation_state=0.)"""

    visualizer.actor_multi_step_rand(n_steps=10)
