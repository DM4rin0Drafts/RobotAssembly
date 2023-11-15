import unittest

import numpy as np
import torch
from torch import Tensor

from constants import IMAGE_WIDTH, IMAGE_HEIGHT
from environment import Tangram, Token


class TestTangram(unittest.TestCase):
    def setUp(self):
        square = Token("square")
        self.tangram = Tangram([square])

    def test_target_and_state_matrices_and_images_not_none_and_correct_shape(self):
        state = self.tangram.state
        self.assertTrue(state is not None)
        self.assertTrue(state.size() == (9, 13))
        self.assertTrue(type(state) == Tensor)

        target_img = self.tangram.target_img
        self.assertTrue(target_img is not None)
        self.assertTrue(target_img.size() == (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertTrue(type(target_img) == Tensor)

        state_img = self.tangram.state_img
        self.assertTrue(state_img is not None)
        self.assertTrue(state_img.size() == (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertTrue(type(state_img) == Tensor)

    def test_state(self):
        token = self.tangram.tokens[0]

        # Temporarily store target center and corners to later test entries of state matrix
        center_target = np.array([-0.5, 0.])
        orientation_target = 0.5
        token.set_pose(center_target, orientation_target)
        corners_target = token.corners

        self.tangram.create_constant_setup(center_target, orientation_target,
                                           center_state=np.array([0.0, 0.]), orientation_state=0.1)
        corners_state = token.corners
        center_state = token.center
        expected_state_unplaced = np.array([[corners_target[0][0], corners_target[0][1], 1, 1],
                                            [corners_target[1][0], corners_target[1][1], 1, 1],
                                            [corners_target[2][0], corners_target[2][1], 1, 1],
                                            [corners_target[3][0], corners_target[3][1], 1, 1],
                                            [corners_state[0][0], corners_state[0][1], 0, 0],
                                            [corners_state[1][0], corners_state[1][1], 0, 0],
                                            [corners_state[2][0], corners_state[2][1], 0, 0],
                                            [corners_state[3][0], corners_state[3][1], 0, 0],
                                            [center_state[0], center_state[1], 0, 0]])
        expected_adj = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 1],
                                 [1, 0, 1, 1, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 1, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 1, 0, 1, 1, 1],
                                 [0, 0, 0, 0, 1, 1, 0, 1, 1],
                                 [0, 0, 0, 0, 1, 1, 1, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 0]])
        expected_unplaced = np.hstack((expected_state_unplaced, expected_adj))
        self.assertTrue(torch.equal(self.tangram.state, Tensor(expected_unplaced)))

        token.move(1, 1, 1)

        corners_new_state = token.corners
        center_new_state = token.center
        expected_state_placed = np.array([[corners_target[0][0], corners_target[0][1], 1, 1],
                                          [corners_target[1][0], corners_target[1][1], 1, 1],
                                          [corners_target[2][0], corners_target[2][1], 1, 1],
                                          [corners_target[3][0], corners_target[3][1], 1, 1],
                                          [corners_new_state[0][0], corners_new_state[0][1], 1, 0],
                                          [corners_new_state[1][0], corners_new_state[1][1], 1, 0],
                                          [corners_new_state[2][0], corners_new_state[2][1], 1, 0],
                                          [corners_new_state[3][0], corners_new_state[3][1], 1, 0],
                                          [center_new_state[0], center_new_state[1], 1, 0]])
        expected_placed = np.hstack((expected_state_placed, expected_adj))
        self.assertTrue(torch.equal(self.tangram.state, Tensor(expected_placed)))


if __name__ == "__main__":
    unittest.main()
