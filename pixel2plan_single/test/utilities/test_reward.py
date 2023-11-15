import unittest

import numpy as np
import torch
from torch import Tensor

from constants import IMAGE_HEIGHT, IMAGE_WIDTH, PATH_PROJECT
from environment import Token, Tangram
from utilities.reward import sparse_reward, dense_reward


class TestReward(unittest.TestCase):
    def setUp(self):
        self.state_img = torch.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        self.target_img = torch.zeros_like(self.state_img)

        square = Token("square")
        self.tangram = Tangram([square])

    def test_sparse_reward_returns_tensor(self):
        reward = sparse_reward(self.state_img, self.target_img)
        self.assertEqual(type(reward), Tensor)

    def test_sparse_reward_perfect(self):
        self.state_img[10:30, 10:30] = 1.
        self.target_img[10:30, 10:30] = 1.
        self.assertEqual(sparse_reward(self.state_img, self.target_img).data, 1.)

    def test_sparse_reward(self):
        self.state_img[8:28, 10:30] = 1.
        self.target_img[10:30, 10:30] = 1.

        n_pixels = self.target_img.sum()
        n_overlapping = n_pixels - 40
        reward = sparse_reward(self.state_img, self.target_img)
        self.assertEqual(reward, n_overlapping / n_pixels)
        self.assertTrue(reward.data <= 1.)
        self.assertTrue(reward.data >= -1.)

    def test_dense_reward_returns_tensor(self):
        self.assertTrue(type(dense_reward(self.tangram.state)), Tensor)

    def test_dense_reward_perfect(self):
        center = np.array([-0.5, 0.])
        orientation = 0.
        self.tangram.create_constant_setup(center_target=center, orientation_target=orientation, center_state=center,
                                           orientation_state=orientation)
        self.assertEqual(dense_reward(self.tangram.state).data, 1.)


if __name__ == "__main__":
    unittest.main()
