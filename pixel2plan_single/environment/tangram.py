from typing import List, Tuple

from constants import *
from environment import Token
from utilities.utilities import *
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np
import os


class Tangram(object):
    def __init__(self, tokens: List[Token], gui=False, constant_setup_only=False, device='cpu'):
        self.device = device

        self.client = None
        self.__initialize_pybullet_simulation(gui)

        self.__tokens = tokens
        for t in self.__tokens:
            t.env = self

        self.__target: torch.Tensor = None
        self.__target_img: torch.Tensor = None

        if constant_setup_only:
            self.create_constant_setup()
        else:
            self.create_random_setup()

    def __initialize_pybullet_simulation(self, gui: bool = False):
        """
        Loads PyBullet client and plane. Additionally sets gravity property and time per simulation step in seconds.
        """
        if gui:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)

        # Set physics parameter
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -10)

        self.client.setTimeStep(SECONDS_PER_STEP)

        # Create game surface
        path = os.path.join(PATH_PROJECT, "environment/urdf/plane.urdf")
        self.client.loadURDF(path)

        self.viewMatrix = self.client.computeViewMatrix(
            cameraEyePosition=[0, 0, 100],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0]
        )

        # set the field of view so that the camera perspective (used for state img) captures
        # the frame that equal the state space between -0.5*STATE_SPACE_SIZE and 0.5*STATE_SPACE_SIZE
        fov = 2 * np.arctan(CAMERA_IMAGE_SIZE / IMAGE_WIDTH) * (180 / np.pi)

        self.projectionMatrix = self.client.computeProjectionMatrixFOV(
            fov=fov,
            aspect=1.0,
            nearVal=99.0,
            farVal=101.0
        )

    def create_constant_setup(self, center_target=(-0.5, 0.), orientation_target=0.,
                              center_state=(0., 0.), orientation_state=0.):
        """
        Sets constant target setup with state matrix, then moves token to a constant position in the simulation again.
        """
        idx = 0
        self.tokens[idx].set_pose(center=center_target, orientation=orientation_target)
        self.__set_target_state(idx=idx)
        self.tokens[idx].set_pose(center=center_state, orientation=orientation_state)

    def create_random_setup(self):
        """
        Sets target setup with state matrix, then randomize token in the simulation again.
        """
        idx = 0
        self.tokens[idx].set_random_pose()
        self.__set_target_state(idx=idx)
        self.tokens[idx].set_random_pose()

    def __set_target_state(self, idx: int):
        target_corners = self.tokens[idx].corners
        self.target = np.array([[*corner, 1, 1] for corner in target_corners])
        self.target_img = self.state_img

    @property
    def tokens(self):
        return self.__tokens

    @property
    def n_token(self):
        return len(self.tokens)

    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, value):
        self.__target = value

    @property
    def target_img(self):
        return self.__target_img

    @target_img.setter
    def target_img(self, value):
        self.__target_img = value

    @property
    def state(self) -> torch.Tensor:
        """
        Represents the state based on:

            Learn2Assemble with Structured Representations and Search for Robotic Architectural Construction,
            Niklas Funk, Georgia Chalvatzaki, Boris Belousov, Jan Peters.
            https://proceedings.mlr.press/v164/funk22a/funk22a.pdf

        The state is represented as a torch tensor of form
                [ s_target   |           ]
            s = [ s_placed   | Adjacency ]
                [ s_unplaced |           ]
        where
            s_target: Target edges as [x, y, placed=True, target=True]
            s_placed: Placed token edges as [x, y, placed=True, target=False]
            s_unplaced: Unplaced token edges and their base as [x, y, placed=False, target=False]
                            -> Base is used to control movements, therefore we want the action for this node
        and a adjacency matrix represented as torch tensor where entry adj[i, j] == 1, iff there is an edge between
        nodes i and j and adj[i, j] == 0 otherwise.

        Note: Nodes are represented by indices beginning from token with index 0. See token_corners for node ordering.
              Also for token that are not targets, the base node is appended after the tokens corners.
        """
        assert self.target is not None, "A target has to be set."

        s_placed = np.empty((0, 4))
        s_unplaced = np.empty((0, 4))
        token = self.tokens[0]
        corners = token.corners
        center = token.center

        if token.is_placed:
            s_placed = np.append(s_placed, np.array([[x, y, 1, 0] for x, y in corners]), axis=0)
            s_placed = np.append(s_placed, np.array([center[0], center[1], 1, 0]).reshape((1, 4)), axis=0)
        else:
            s_unplaced = np.append(s_unplaced, np.array([[x, y, 0, 0] for x, y in corners]), axis=0)
            s_unplaced = np.append(s_unplaced, np.array([center[0], center[1], 0, 0]).reshape((1, 4)), axis=0)

        state_no_adj = np.vstack((self.target, s_placed, s_unplaced))

        n_nodes = state_no_adj.shape[0]
        adj = np.zeros((n_nodes, n_nodes))
        target_idx = [i for i in range(len(corners))]  # Indices in adjacency matrix for target nodes
        token_idx = [i for i in range(len(corners), len(corners) * 2 + 1)]
        control_idx = -1  # Index of control node
        adj[target_idx, control_idx] = 1
        adj[control_idx, target_idx] = 1
        adj[tuple(np.meshgrid(target_idx, target_idx))] = 1
        adj[tuple(np.meshgrid(token_idx, token_idx))] = 1
        adj -= np.diag(np.diag(adj))

        state = np.hstack((state_no_adj, adj))
        return torch.as_tensor(state).to(self.device).float()

    def overwrite_target_in_state(self, in_state, overwrite_state):
        # TODO: #9 Generalize to multiple Tokens
        idx = 0
        n_corners = len(self.tokens[idx].corners)
        out_state = in_state.clone()
        out_state[:n_corners, :2] = overwrite_state[n_corners:n_corners * 2, :2]
        return out_state

    @property
    def state_img(self) -> torch.Tensor:
        """ Returns state as binary image. """
        _, _, img_rgb, _, _ = self.client.getCameraImage(
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix
        )
        img_binary = rgb2binary(img_rgb)
        return numpy2torch(img_binary)

    def run(self):
        """ Starts real time simulation for human interaction """
        self.client.setRealTimeSimulation(1)

    def stop(self):
        """ Stops real time simulation """
        self.client.setRealTimeSimulation(0)

    def reset_tokens(self):
        """ Resets token to default positions """
        for token in self.tokens:
            token.reset()

    def __del__(self):
        self.client.disconnect()

    def action_from_data(self, vx: float, vy: float, wz: float):
        data = np.asarray([vx, vy, wz]).astype(np.float32)
        return torch.from_numpy(data).to(self.device)
