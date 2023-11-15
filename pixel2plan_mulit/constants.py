# Pybullet simulation settings
# Determines time of each simulation step
SECONDS_PER_STEP = 1 / 240
# Number of calls to stepSimulation when moving a token
SIMULATION_STEPS = 240  # 240 steps = 1 second

TOKEN_SPAWN_REGION_SIZE = 2  # [-1, 1]
STATE_SPACE_SIZE = 4  # [-2, 2]
CAMERA_IMAGE_SIZE = 5  # [-2.5, 2.5]
HIDDEN_POSITION = 2 * STATE_SPACE_SIZE

BASE_POSITIONS = ([6, 9, 0],
                  [4, 9, 0],
                  [2, 9, 0],
                  [0, 9, 0],
                  [-2, 9, 0],
                  [-4, 9, 0],
                  [-6, 9, 0])

# Constants used to get state as rgb image
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200

# Defines corner coordinates of each game token with base at origin and no rotation and without scaling.
# World Coordinate System:
#   y
#   ^
#   |
#   z - - > x
#
# square:        parallelogram:        triangle:
# c2 ---- c3          c2 ---- c3           c2
# |       |          /       /            /   \
# |       |        /       /             /     \
# c1 ---- c4     c1 ---- c4            c1 ---- c3
# The corner coordinates are given as [c1, c2, c3, c4] for parallelogram and square and [c1, c2, c3] for triangles.
# Note: All points correspond to a scale in metres while scaling is applied when loaded into simulation.
# The z-component is given as highest point.
Z = 0.006  # TODO: Is the z-coordinate even needed?
CORNERS_IN_URDF_BASE_AT_ORIGIN_NO_ROTATION = {
    "square": [(-0.020, -0.020, Z),
               (-0.020, 0.020, Z),
               (0.020, 0.020, Z),
               (0.020, -0.020, Z)],
    "parallelogram": [(-0.0405814743041992, -0.0130814752578735, Z),
                      (-0.0144185247421265, 0.0130814752578735, Z),
                      (0.0405814743041992, 0.0130814752578735, Z),
                      (0.0144185247421265, -0.0130814752578735, Z)],
    "triangle1": [(-0.0565685463561217488928658, -0.0282842712402344, Z),
                  (0.0, 0.0282842712402344, Z),
                  (0.0565685386048157511071342, -0.0282842712402344, Z)],
    "triangle2": [(-0.0388908729553223, -0.0194454364776611, Z),
                  (0.0, 0.0194454402923584, Z),
                  (0.0388908767700195, -0.0194454364776611, Z)],
    "triangle3": [(-0.0282842693034356860169766, -0.0141421346664429, Z),
                  (0.0, 0.0141421365737915, Z),
                  (0.0282842731770330139830234, -0.0141421346664429, Z)]

}

# Scaling applied in .urdf
SCALING = [10, 10, 50]
