import torch
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
    def __init__(self, tokens: List[Token], gui=False, device='cpu'):
        self.device = device

        self.client = None
        self.__initialize_pybullet_simulation(gui)

        self.__tokens = tokens
        self.target_center = None

        for t in self.__tokens:
            t.env = self

        self.__target: torch.Tensor = None
        self.__target_img: torch.Tensor = None

        self.__tokens_index = self.token_id()
        self.__control_nodes = None
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
        path = os.path.join(os.getcwd(), "environment/urdf/plane.urdf")
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

    def create_random_setup(self):
        """
        Sets a random target and spawns token. Random target is sampled from a 2D Gaussian. Set target setup
        with state matrix, then randomize token in the simulation again
        """
        self.target = None

        for i in range(2):
            self.hide_tokens()

            for token in self.tokens:
                self.rand_position_generator(token.ID)

            if i == 0:
                self.set_target_state()

    def set_target_state(self):
        """
        Set the target state and target image of the spawned tokens in the environment.
        """
        lines = np.array([len(token.corners) for token in self.tokens]).sum()
        state_target = self.state.cpu().numpy()

        self.target = state_target[:lines, :4]
        self.target_img = self.state_img

    def rand_position_generator(self, id):
        while True:
            self.tokens[self.token_idx[id]].set_random_pose()

            result = self.check_collision(id)
            if result:
                break

    def check_collision(self, id):
        token_ids = [token.ID for token in self.tokens]
        token_ids.remove(id)

        results = []
        for t_id in token_ids:
            results.append(self.object_collision(t_id, id))

        if sum(results) == 0:
            # no collision detected
            return True
        else:
            return False

    def object_collision(self, body1, body2, distance=0.0):
        """
        Check collision of two objects in the pybullet simulation and visualize it in the simulation

        Parameters
        ----------
        body1: int, required
            body unique id, as returned by loadURDF etc
        body2: int, required
            body unique id, as returned by loadURDF etc
        distance: float, optional
            Maximum distance of a link collision

        return
        ----------
        If collision, return true

        """
        collision_results = list(
                p.getClosestPoints(bodyA=body1,
                                   bodyB=body2,
                                   distance=distance,
                                   physicsClientId=0))

        if len(collision_results) == 0:
            return False  # no_collision
        else:
            return True

    def token_id(self):
        """
        Creates a dictionary of all the token ids. this function return the indes of the needed token id
        """

        token_ids = dict()

        for idx, token in enumerate(self.tokens):
            token_ids[token.ID] = idx

        return token_ids

    @property
    def token_idx(self):
        """
        Return a dictionary of the token ids of the equivalent indexesx for self.tokens
        """
        return self.__tokens_index

    @property
    def control_nodes(self):
        return self.__control_nodes

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

    def generate_adjacency_matrix(self, state_length):
        """
        Args:
            state_length: length of the current generated state matrix without the adjacency matrix
                          input is the length of the state matrix as an integer

        a adjacency matrix represented as torch tensor where entry adj[i, j] == 1, iff there is an edge between
        nodes i and j and adj[i, j] == 0 otherwise.

        Note: Nodes are represented by indices beginning from token with index 0. See token_corners for node ordering.
              Also for token that are not targets, the base node is appended after the tokens corners
        """

        adj = np.zeros((state_length, state_length))

        m_shift = 0
        length = sum(self.n_corners)

        for idx, corners in enumerate(self.n_corners):
            # target self conection
            connection = np.ones((corners, corners))
            x, y = sum(self.n_corners[:idx]), sum(self.n_corners[:idx]) + corners
            adj[x:y, x:y] = connection

            #target with control node self connection
            connection = np.ones((corners + 1, corners + 1))
            x = length + x + m_shift
            y = length + y + 1 +  m_shift
            adj[x:y, x:y] = connection

            m_shift += 1

            # add target corner to control node corners
            adj[y - 1, 0:length] = np.ones(length)
            adj[0:length, y - 1] = np.ones(length)

        np.fill_diagonal(adj, 0)

        return adj


    def add_center_to_state(self, id, s, is_target):
        cx, cy = self.tokens[id].center

        if is_target:
            s = np.append(s, np.array([cx, cy, 1, 1]).reshape((1, 4)), axis=0)
        elif self.tokens[id].is_placed:
            s = np.append(s, np.array([cx, cy, 1, 0]).reshape((1, 4)), axis=0)
        elif not self.tokens[id].is_placed:
            s = np.append(s, np.array([cx, cy, 0, 0]).reshape((1, 4)), axis=0)

        # wir benötigen die anzahl der corners der token nur einmal
        #self.n_corners.append(len(CORNERS_IN_URDF_BASE_AT_ORIGIN_NO_ROTATION[self.tokens[id].name]))

        return s

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
        self.n_corners = [len(CORNERS_IN_URDF_BASE_AT_ORIGIN_NO_ROTATION[token.name]) for token in self.tokens]
        #self.n_corners = []
        state = np.empty((0, 4))
        self.__control_nodes = []

        j = 0
        for add_control_node in [False, True]:
            for idx, token in enumerate(self.tokens):
                s = np.empty((0, 4))
                corners = token.corners

                if add_control_node:
                    if token.is_placed:
                        s = np.append(s, np.array([[c[0], c[0], 1, 0] for c in corners]), axis=0)
                    else:
                        s = np.append(s, np.array([[c[0], c[0], 0, 0] for c in corners]), axis=0)
                    is_target = False

                else:
                    if isinstance(self.target, np.ndarray):
                        x, y = sum(self.n_corners[:idx]), sum(self.n_corners[:idx]) + len(corners)
                        s = self.target[x:y]
                    else:
                        s = np.array([[c[0], c[0], 1, 1] for c in corners])
                    is_target = True

                state = np.append(state, s, axis=0)

                if add_control_node:
                    state = self.add_center_to_state(idx, state, is_target)
                    self.__control_nodes.append(len(state) - 1)

        if isinstance(self.target, np.ndarray):
            # return state with adjacency
            adjacency_matrix = self.generate_adjacency_matrix(state.shape[0])
            state = np.hstack((state, adjacency_matrix))

            return torch.as_tensor(state).to(self.device).float()
        else:
            # Return target state for initialization
            return torch.as_tensor(state).to(self.device).float()

    def overwrite_target_in_state(self, in_state, overwrite_state):
        # TODO optimize not efficient for deleting rows in tensor  --> do it like that a = a[torch.arange(a.size(0))!=1]
        lines = np.sum(np.array(self.n_corners))

        out_state = in_state.clone()
        overwrite_state = torch.tensor(np.delete(overwrite_state.cpu().numpy(), self.control_nodes, 0)[lines:, :2])

        out_state[:lines, :2] = overwrite_state
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

    def hide_tokens(self):
        for token in self.tokens:
            token.hide()

    def __del__(self):
        self.client.disconnect()

    def action_from_data(self, vx: float, vy: float, wz: float):
        data = np.asarray([vx, vy, wz]).astype(np.float32)
        return torch.from_numpy(data).to(self.device)
