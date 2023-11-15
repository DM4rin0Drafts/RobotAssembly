import os
import time
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
import torch

from constants import actor_structure, critic_structure
from environment.tangram import Tangram
from environment.token import Token
from models.actor_mha import ActorMHA, actor_mha_from
from models.critic_mha import CriticMHA, critic_mha_from
from models.architectures.multi_head_attention import MPNNMultidimFullAttention
from utilities.reward import sparse_reward, dense_reward
from utilities.utilities import rgb2binary, rgb2grayscale


def sigmoid(x: float):
    return np.maximum(0.01, 1 / (1 + np.exp(0.001 * x)))


def create_random_setup(tokenlist: list = None, n: int = 10000, sleep: float = 3):
    if tokenlist is None:
        tokenlist = ["triangle2", "square"]
    if type(tokenlist) != list:
        tokenlist = [tokenlist]

    tokens = [Token(name=t) for t in tokenlist]
    tangram = Tangram(tokens=tokens, gui=True)
    for i in range(n):
        state_target = tangram.create_random_setup(0.0, sigmoid(i))
        print("Iteration {}: {}".format(i, state_target))
        time.sleep(sleep)


def draw_token_edges(name: str = None):
    if name is None:
        name = "square"
    token = Token(name=name)
    tangram = Tangram(tokens=[token], gui=True)

    tangram.tokens[0].center = (0.5, 1.3)
    tangram.tokens[0].orientation = 0.23

    corners = tangram.tokens[0].corners
    print(corners)
    if len(corners) == 4:
        c1, c2, c3, c4 = corners
        f = [c1[0], c1[1], c1[2] / 3]
        t = [c2[0], c2[1], c2[2] / 3]
        tangram.client.addUserDebugLine(f, t, lineColorRGB=[1, 0, 0], lineWidth=1)  # red

        f = [c1[0], c1[1], c1[2] / 3]
        t = [c4[0], c4[1], c4[2] / 3]
        tangram.client.addUserDebugLine(f, t, lineColorRGB=[1, 1, 0], lineWidth=1)  # yellow

        f = [c2[0], c2[1], c2[2] / 3]
        t = [c3[0], c3[1], c3[2] / 3]
        tangram.client.addUserDebugLine(f, t, lineColorRGB=[1, 0, 1], lineWidth=1)  # purple

        f = [c3[0], c3[1], c3[2] / 3]
        t = [c4[0], c4[1], c4[2] / 3]
        tangram.client.addUserDebugLine(f, t, lineColorRGB=[1, 1, 1], lineWidth=1)  # white
    else:
        c1, c2, c3 = corners
        f = [c1[0], c1[1], c1[2] / 3]
        t = [c2[0], c2[1], c2[2] / 3]
        tangram.client.addUserDebugLine(f, t, lineColorRGB=[1, 0, 0], lineWidth=1)  # red

        f = [c1[0], c1[1], c1[2] / 3]
        t = [c3[0], c3[1], c3[2] / 3]
        tangram.client.addUserDebugLine(f, t, lineColorRGB=[1, 1, 0], lineWidth=1)  # yellow

        f = [c2[0], c2[1], c2[2] / 3]
        t = [c3[0], c3[1], c3[2] / 3]
        tangram.client.addUserDebugLine(f, t, lineColorRGB=[1, 0, 1], lineWidth=1)  # purple

    time.sleep(10000)


def print_joints_and_links_of_token_with_name(name: str = "square"):
    def resolve_joint_type(type: int):
        if type == 0:
            return "continuous"
        elif type == 1:
            return "prismatic"
        else:
            return "unknown"

    token = Token(name=name)
    tangram = Tangram(tokens=[token], gui=True)

    tangram.tokens[0].center = (0, 0)
    tangram.tokens[0].orientation = 0

    bodyId = 1  # id of object in simulation
    n_joints = tangram.client.getNumJoints(bodyUniqueId=bodyId)
    print("--- TOKEN INFO ({}) ---".format(name))
    for i in range(n_joints):
        joint_info = tangram.client.getJointInfo(bodyUniqueId=bodyId, jointIndex=i)
        shape_data = tangram.client.getCollisionShapeData(objectUniqueId=bodyId, linkIndex=i)
        print("JOINT")
        print("\tindex: ", joint_info[0])
        print("\tname: ", joint_info[1])
        print("\ttype: ", resolve_joint_type(joint_info[2]))
        if i == n_joints - 1:
            print()
            print("\tLINK WITH COLLISION SHAPE")
            shape_data = shape_data[0]
            print("\t\tindex: ", shape_data[1])
            print("\t\tgeometry type: mesh")  # We only use mesh type
            print("\t\tscaling factor: ", shape_data[3])
            print("\t\tfile: ", shape_data[4])
        print("---")
    print("-----------------------")


def plot_token(name: str = "square"):
    xyz = None
    if name == "square":
        xyz = "-20 -20 0 -20 -20 6 20 -20 0 20 -20 6 20 -20 0 20 -20 6 20 20 0 20 20 6 20 20 0 20 20 6 -20 20 0 -20 " \
              "20 6 -20 20 0 -20 20 6 -20 -20 0 -20 -20 6 -20 -20 0 20 -20 0 20 20 0 -20 20 0 -20 -20 6 20 -20 6 20 " \
              "20 6 -20 20 6".split(" ")
    elif name == "parallelogram":
        xyz = "-14.4185247421265 13.0814752578735 0 -14.4185247421265 13.0814752578735 6 -40.5814743041992 " \
              "-13.0814752578735 0 -40.5814743041992 -13.0814752578735 6 -40.5814743041992 -13.0814752578735 0 " \
              "-40.5814743041992 -13.0814752578735 6 14.4185247421265 -13.0814752578735 0 14.4185247421265 " \
              "-13.0814752578735 6 14.4185247421265 -13.0814752578735 0 14.4185247421265 -13.0814752578735 6 " \
              "40.5814743041992 13.0814752578735 0 40.5814743041992 13.0814752578735 6 40.5814743041992 " \
              "13.0814752578735 0 40.5814743041992 13.0814752578735 6 -14.4185247421265 13.0814752578735 0 " \
              "-14.4185247421265 13.0814752578735 6 -14.4185247421265 13.0814752578735 0 -40.5814743041992 " \
              "-13.0814752578735 0 14.4185247421265 -13.0814752578735 0 40.5814743041992 13.0814752578735 0 " \
              "-14.4185247421265 13.0814752578735 6 -40.5814743041992 -13.0814752578735 6 14.4185247421265 " \
              "-13.0814752578735 6 40.5814743041992 13.0814752578735 6".split(" ")
    elif name == "triangle1":
        xyz = "56.5685386657715 -28.2842712402344 0 56.5685386657715 -28.2842712402344 6 6.09557488928658E-08 " \
              "28.2842712402344 0 6.09557488928658E-08 28.2842712402344 6 6.09557488928658E-08 28.2842712402344 0 " \
              "6.09557488928658E-08 28.2842712402344 6 -56.568546295166 -28.2842712402344 0 -56.568546295166 " \
              "-28.2842712402344 6 -56.568546295166 -28.2842712402344 0 -56.568546295166 -28.2842712402344 6 " \
              "56.5685386657715 -28.2842712402344 0 56.5685386657715 -28.2842712402344 6 56.5685386657715 " \
              "-28.2842712402344 0 6.09557488928658E-08 28.2842712402344 0 -56.568546295166 -28.2842712402344 0 " \
              "56.5685386657715 -28.2842712402344 6 6.09557488928658E-08 28.2842712402344 6 -56.568546295166 " \
              "-28.2842712402344 6".split(" ")
    elif name == "triangle2":
        xyz = "38.8908767700195 -19.4454364776611 0 38.8908767700195 -19.4454364776611 6 0 19.4454402923584 0 0 " \
              "19.4454402923584 6 0 19.4454402923584 0 0 19.4454402923584 6 -38.8908729553223 -19.4454364776611 0 " \
              "-38.8908729553223 -19.4454364776611 6 -38.8908729553223 -19.4454364776611 0 -38.8908729553223 " \
              "-19.4454364776611 6 38.8908767700195 -19.4454364776611 0 38.8908767700195 -19.4454364776611 6 " \
              "38.8908767700195 -19.4454364776611 0 0 19.4454402923584 0 -38.8908729553223 -19.4454364776611 0 " \
              "38.8908767700195 -19.4454364776611 6 0 19.4454402923584 6 -38.8908729553223 -19.4454364776611 6" \
            .split(" ")
    elif name == "triangle3":
        xyz = "28.284273147583 -14.1421346664429 0 28.284273147583 -14.1421346664429 6 -2.94500139830234E-08 " \
              "14.1421365737915 0 -2.94500139830234E-08 14.1421365737915 6 -2.94500139830234E-08 14.1421365737915 0 " \
              "-2.94500139830234E-08 14.1421365737915 6 -28.2842693328857 -14.1421346664429 0 -28.2842693328857 " \
              "-14.1421346664429 6 -28.2842693328857 -14.1421346664429 0 -28.2842693328857 -14.1421346664429 6 " \
              "28.284273147583 -14.1421346664429 0 28.284273147583 -14.1421346664429 6 28.284273147583 " \
              "-14.1421346664429 0 -2.94500139830234E-08 14.1421365737915 0 -28.2842693328857 -14.1421346664429 0 " \
              "28.284273147583 -14.1421346664429 6 -2.94500139830234E-08 14.1421365737915 6 -28.2842693328857 " \
              "-14.1421346664429 6".split(" ")
    else:
        print("Token name {} unkown.".format(name))

    x, y, z = [], [], []
    i = 0
    while i < len(xyz):
        x.append(float(xyz[i]))
        y.append(float(xyz[i + 1]))
        z.append(float(xyz[i + 2]))
        i += 3

    x1, y1, x2, y2 = [], [], [], []
    for i in range(len(z)):
        if z[i] == 0:
            x1.append(x[i])
            y1.append(y[i])
        else:
            x2.append(x[i])
            y2.append(y[i])

    print("x1: ", x1)
    print("y1: ", y1)
    print()
    print("x2: ", x2)
    print("y2: ", y2)
    plt.figure()

    plt.plot(x1, y1, 'bo')
    plt.plot(x2, y2, 'ro')

    plt.show()


def plot_tokens_aligned(savefig=False):
    tokenlist = ["square", "parallelogram", "triangle1", "triangle2", "triangle3"]
    tokens = [Token(name=t) for t in tokenlist]
    tangram = Tangram(tokens=tokens, gui=True)

    tangram.client.resetDebugVisualizerCamera(3, 30, -90, [0, 0, 0])

    tangram.tokens[0].center = (-0.5, 0.5)
    tangram.tokens[0].orientation = 0
    tangram.tokens[1].center = (0.5, 0.5)
    tangram.tokens[1].orientation = 0
    tangram.tokens[2].center = (-0.75, 0)
    tangram.tokens[2].orientation = 0
    tangram.tokens[3].center = (0, 0)
    tangram.tokens[3].orientation = 0
    tangram.tokens[4].center = (0.75, 0)
    tangram.tokens[4].orientation = 0

    rgb = tangram.state_img.squeeze()
    plt.figure()
    plt.imshow(rgb, cmap="gray")
    plt.axis("off")
    if savefig:
        plt.savefig("before_move_rgb.svg", format='svg', dpi=1200)
    plt.show()

    time.sleep(10000)


def test_actor_critic(tangram):
    n_features_actor = 4
    dim_single_pred_actor = 3
    n_features_critic = n_features_actor + dim_single_pred_actor
    dim_single_pred_critic = 1

    n_layers = 3
    embed_dim = 64
    n_heads = 4
    n_hidden_readout = []

    actor = ActorMHA(n_features_actor, n_layers, embed_dim, n_heads, n_hidden_readout, dim_single_pred_actor)
    # Action is added to state, therefore the number of features is increased by 3 * n_token
    critic = CriticMHA(n_features_critic, n_layers, embed_dim, n_heads, n_hidden_readout, dim_single_pred_critic)


def main():
    path = os.path.join(os.getcwd(), "runs/Sep05_12-18-36_x")
    n1 = torch.load(os.path.join(path, "data0.tar"))
    n2 = torch.load(os.path.join(path, "data1.tar"))

    actor_state1 = n1["Actor"]
    critic_state1 = n1["Critic"]

    actor_state2 = n2["Actor"]
    critic_state2 = n2["Critic"]

    actor1 = ActorMHA(actor_structure.n_features,
                      actor_structure.n_layers,
                      actor_structure.embed_dim,
                      actor_structure.n_heads,
                      actor_structure.n_hidden_readout,
                      actor_structure.dim_single_pred,
                      0,
                   "cuda:0")
    actor2 = actor_mha_from(actor1)

    actor1.state = actor_state1
    actor2.state = actor_state2

    parameters1 = list(actor1.parameters())[0].detach().clone().cpu().numpy()
    parameters2 = list(actor2.parameters())[0].detach().clone().cpu().numpy()
    a = parameters1 == parameters2
    print(len(np.where(a == False)[0]))

    critic1 = CriticMHA(critic_structure.n_features,
                        critic_structure.n_layers,
                        critic_structure.embed_dim,
                        critic_structure.n_heads,
                        critic_structure.n_hidden_readout,
                        critic_structure.dim_single_pred,
                        0,
                     "cuda:0")
    critic2 = critic_mha_from(critic1)

    critic1.state = critic_state1
    critic2.state = critic_state2

    parameters1 = list(critic1.parameters())[0].detach().clone().cpu().numpy()
    parameters2 = list(critic2.parameters())[0].detach().clone().cpu().numpy()
    a = parameters1 == parameters2
    print(len(np.where(a == False)[0]))


if __name__ == "__main__":
    main()
