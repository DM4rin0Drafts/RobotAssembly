import torch
import numpy as np
from torch import Tensor

from constants import TOKEN_SPAWN_REGION_SIZE


def sparse_reward(state_img: Tensor, target_img: Tensor) -> Tensor:
    """ Computes reward as sum of overlapping pixels between binary images of state and target. Uses normalization
        if token is specified by name to account for different sizes of tokens. """
    assert state_img.size() == target_img.size(), "Error: Inputs have to be of same size but where {} and {}.".format(
        state_img.size(), target_img.size()
    )
    # TODO: For the future: For multiple targets we might have to solve the problem, that token1 covers the pixels
    #  of the target of token2. This wrongly increases the reward because we have overlapping pixels.
    #  One solution might be: Sample target_imgs for each target (e.g. by making other targets invisible) and do this
    #  for all targets. At the end calculate a reward that accounts for all targets
    state_pixels = state_img.squeeze()
    target_pixels = target_img.squeeze()
    n_overlapping = target_pixels[state_pixels == 1].sum()
    n_target = target_pixels.sum()

    if n_target != 0:
        return 2 * (n_overlapping / n_target) - 1
    else:
        return torch.tensor(-1.0)


def dense_reward(state: Tensor) -> Tensor:
    target_corners = state[:4, :2].cpu()
    target_center = calc_center(target_corners)
    token_center = state[-1, :2].cpu()

    distance = torch.norm(target_center - token_center)

    return distance_activation(distance)


def distance_activation(x, log=True):
    lowest_point = np.sqrt(2 * 
     ** 2)
    lowest_activation = -1
    if log:
        s = 0.01  # the lower, the steeper the curve falls of
        activation = (-np.log(x + s) + np.log(lowest_point + s)) / (-np.log(s) + np.log(lowest_point + s)) * \
                     (1 - lowest_activation) + lowest_activation
    else:  # linear
        activation = (lowest_activation - 1) / lowest_point * x + 1

    return max(activation, torch.tensor(-1.0))


def calc_center(pts):
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]
    p4 = pts[3]

    x_diff = torch.tensor([p1[0] - p3[0], p2[0] - p4[0]])
    y_diff = torch.tensor([p1[1] - p3[1], p2[1] - p4[1]])

    div = torch.det(torch.vstack([x_diff, y_diff]))

    d = torch.tensor([torch.det(torch.vstack([p1, p3])), torch.det(torch.vstack([p2, p4]))])

    x = np.linalg.det(torch.vstack([d, x_diff])) / div
    y = np.linalg.det(torch.vstack([d, y_diff])) / div

    return torch.tensor([x, y])
