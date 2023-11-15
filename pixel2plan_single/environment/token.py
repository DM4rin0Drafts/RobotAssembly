from constants import *
from typing import Tuple
import os
import pybullet as p
import numpy as np


class Token(object):
    def __init__(self, name, env=None, center=(0.0, 0.0), orientation=0.0):
        # TODO: Consider using shape enums for identification
        self.name = name
        self.__ID = None
        self.__env = env

        self.__center_at_spawn = center
        self.__orientation_at_spawn = orientation

        self.__is_placed = False

    @property
    def path(self):
        path = os.path.join(PATH_PROJECT, f"environment/urdf/{self.name}.urdf")
        if not os.path.exists(path):
            raise RuntimeError(f'File {path} does not exist.')
        return path

    def _load_urdf(self):
        self.__ID = self.client.loadURDF(fileName=self.path,
                                         basePosition=[0, 0, 0],
                                         baseOrientation=[0, 0, 0, 1],
                                         useFixedBase=True)

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, value):
        self.__env = value
        self._load_urdf()

    @property
    def client(self):
        return self.env.client

    @property
    def ID(self):
        return self.__ID

    @property
    def center_at_spawn(self):
        return self.__center_at_spawn

    @center_at_spawn.setter
    def center_at_spawn(self, x_y_z):
        self.__center_at_spawn = x_y_z

    @property
    def orientation_at_spawn(self):
        return self.__orientation_at_spawn

    @orientation_at_spawn.setter
    def orientation_at_spawn(self, phi):
        self.__orientation_at_spawn = phi

    @property
    def center(self):
        center_xyz, _ = self.client.getBasePositionAndOrientation(bodyUniqueId=self.ID)
        center_xy = center_xyz[:2]
        return center_xy

    @center.setter
    def center(self, center):
        """
        Sets the center point position of the Token
        Args:
            center: (x,y)-position of the Token; each (x,y) in [-0.5 * STATE_SPACE_SIZE, 0.5 * STATE_SPACE_SIZE]
        """

        center = tuple(np.clip(center, -0.5 * STATE_SPACE_SIZE, 0.5 * STATE_SPACE_SIZE))

        x, y = center
        phi = self.orientation
        orientation_quat = p.getQuaternionFromEuler([0, 0, phi])
        self.client.resetBasePositionAndOrientation(bodyUniqueId=self.ID,
                                                    posObj=[x, y, 0],
                                                    ornObj=orientation_quat)

    @property
    def orientation(self):
        _, orientation_quat = self.client.getBasePositionAndOrientation(bodyUniqueId=self.ID)
        phi = p.getEulerFromQuaternion(orientation_quat)[-1]
        return phi

    @orientation.setter
    def orientation(self, phi):
        """
        Sets the orientation of the Token
        Args:
            phi: Rotation angle of the Token in [-pi, pi]
        """

        assert -np.pi <= phi <= np.pi

        x, y = self.center
        orientation_quat = p.getQuaternionFromEuler([0, 0, phi])
        # TODO: Check if the client calls can be made from the env directly
        #  (e.g. by accessing all the tokens positions and orientations)
        self.client.resetBasePositionAndOrientation(bodyUniqueId=self.ID,
                                                    posObj=[x, y, 0],
                                                    ornObj=orientation_quat)

    @property
    def is_placed(self):
        return self.__is_placed

    @is_placed.setter
    def is_placed(self, placed):
        self.__is_placed = placed

    def move(self, rel_vx: float, rel_vy: float, rel_wz: float):
        """ Moves token according to given velocities.

        Args:
            rel_vx: Relative velocity x-direction
                    e.g. rel_vx=1 means move all the way from the left to all the way to the right of the state space
                    this ensures that with 1 move all possible states can be reached
            rel_vy: Relative velocity y-direction
            rel_wz: Relative angular velocity z-axis
        """

        assert all((-1 <= val <= 1) for val in [rel_vx, rel_vy, rel_wz])

        vx = TOKEN_SPAWN_REGION_SIZE * rel_vx
        vy = TOKEN_SPAWN_REGION_SIZE * rel_vy
        wz = np.pi * rel_wz

        for _ in range(SIMULATION_STEPS):
            self.client.resetBaseVelocity(self.ID, [vx, vy, 0], [0, 0, wz])
            self.client.stepSimulation()
        self.fix_position()

    def set_pose(self, center: Tuple, orientation: float):
        self.center = center
        self.orientation = orientation

    def set_random_pose(self):
        """
        Sets the center and the orientation of the Token randomly.
        The center is uniformely sampled in [-0.5 * TOKEN_SPAWN_REGION_SIZE, 0.5 * TOKEN_SPAWN_REGION_SIZE]
        The orientation is uniformely sampled in [-pi, pi]
        Use uniform instead of normal initialization to avoid high percentage at borders -1 and 1
        """

        self.center = 0.5 * TOKEN_SPAWN_REGION_SIZE * np.random.uniform(-1, 1, size=2)
        self.orientation = np.pi * np.random.uniform(-1, 1)

    def fix_position(self):
        self.client.resetBaseVelocity(self.ID, [0, 0, 0], [0, 0, 0])
        self.__is_placed = True

    def reset(self):
        x, y = self.__center_at_spawn
        orientation_quat = p.getQuaternionFromEuler([0, 0, self.__orientation_at_spawn])
        self.client.resetBasePositionAndOrientation(bodyUniqueId=self.ID,
                                                    posObj=[x, y, 0],
                                                    ornObj=orientation_quat)
        self.__is_placed = False

    @property
    def corners(self):
        """
        Returns the x- and y-coordinate and maximal z-coordinate (of collision) of the corner positions of a token in
        world coordinates.

        World Coordinate System:
        y
        ^
        |
        z - - > x

        square:        parallelogram:        triangle:
        c2 ---- c3          c2 ---- c3            c2
        |       |          /       /             /  \
        |       |        /       /             /     \
        c1 ---- c4     c1 ---- c4            c1 ---- c3

        Returns:
            corners: Numpy array of corners in order as described above. 4 elements if no triangle, 3 else.
        """
        x, y = self.center
        phi = self.orientation

        rot_trans_matrix = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                                     [np.sin(phi), np.cos(phi), 0, y],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])

        # TODO: Idea: Since we only work with x and y coordinates we can do these simple
        #  transformations (scaling, rotation, translation) without a matrix and just by manipulating x and y
        # 4th corner is optional as triangles only have 3 corners
        corners = CORNERS_IN_URDF_BASE_AT_ORIGIN_NO_ROTATION[self.name]
        corners_xy = []
        for corner in corners:
            corner = np.array(corner).reshape((-1, 1))  # transform to column vector
            scaling = np.array(SCALING).reshape((-1, 1))  # transform to column vector
            corner_scaled = corner * scaling
            corner_hom = np.vstack((corner_scaled, 1))
            corner_hom = rot_trans_matrix @ corner_hom
            corner_xy = corner_hom.reshape((-1))[:2]
            corners_xy.append(tuple(corner_xy))
        return corners_xy
