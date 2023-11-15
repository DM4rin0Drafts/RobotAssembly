import os
import csv

import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from torch.utils.tensorboard import SummaryWriter
from models.replaybuffer import ReplayBuffer
from models.critic_mha import Critic
from models.actor_mha import Actor
from environment.tangram import Tangram
from utilities.ddpg import DDPG
from utilities.reward import reward
from utilities.utilities import *
from itertools import permutations


class Master(object):

    def __init__(self, args, demo=False, device="cpu"):

        self.demo = demo
        self.device = device

        self.n_episodes = args.n_episodes
        self.n_transitions = args.n_transitions

        self.env = Tangram(tokens=args.tokens, gui=args.gui, device=device)
        self.writer = SummaryWriter()

        self.actor = Actor(n_features=args.n_features_actor,
                           n_layers=args.n_layers,
                           embed_dim=args.embed_dim,
                           n_heads=args.n_heads,
                           n_hidden_readout=args.n_hidden_readout,
                           dim_single_pred=args.dim_single_pred_actor,
                           device=device)

        self.critic = Critic(n_features=args.n_features_critic,
                             n_layers=args.n_layers,
                             embed_dim=args.embed_dim,
                             n_heads=args.n_heads,
                             n_hidden_readout=args.n_hidden_readout,
                             dim_single_pred=args.dim_single_pred_critic,
                             control_nodes=self.env.control_nodes,
                             device=device)

        self.writer.add_graph(self.actor, torch.zeros(9, 13).to(self.device))

        self.replaybuffer = ReplayBuffer(buffer_size=args.buffer_size, device=device)
        if not self.demo:
            self.ddpg = DDPG(self.actor, self.critic,
                             lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                             gamma=args.gamma, tau=args.tau)

        # Load state
        self.load(args.load)

        self.finished_jobs = 0
        self.given_jobs = 0

    @property
    def n_token(self):
        return self.env.n_token

    @property
    def actor_params(self):
        return self.actor.params

    @property
    def critic_params(self):
        return self.critic.params

    def iteration(self):
        pass

    @staticmethod
    def plot(init_states, predictions, targets) -> plt.Figure:
        fig, axs = plt.subplots(len(init_states), 4)

        for i in range(len(init_states)):
            axs[i, 0].imshow(init_states[i], "gray")
            axs[i, 1].imshow(predictions[i], "gray")
            axs[i, 2].imshow(targets[i], "gray")
            axs[i, 3].imshow((predictions[i] + targets[i]).float() / 2, "gray")

            for j in range(len(axs[i])):
                axs[i, j].axis("off")

        axs[0, 0].set_title("Init")
        axs[0, 1].set_title("Prediction")
        axs[0, 2].set_title("Target")
        axs[0, 3].set_title("P/T-Overlap")

        return fig

    def show_result(self, result, target):
        self.plot(result, target)
        plt.show()

    def save_result(self, init_states, predictions, targets, t_step):
        fig = self.plot(init_states, predictions, targets)
        self.writer.add_figure(f"Episode {t_step} - Top {len(init_states)} Results", fig, t_step)

    def load_target(self, _path):
        target_img_rgb = plt.imread(_path)
        target_img_binary = rgb2binary(target_img_rgb)
        self.env.target_img = target_img_rgb
        self.env.target = numpy2torch(target_img_binary)

    def run(self):
        while self.finished_jobs < self.n_episodes:
            self.iteration()

    def save(self, run=None):
        if run is None:
            run = self.writer.get_logdir()
        _run = run

        _path = os.path.join(os.getcwd(), f'{_run}/data.tar')
        print(f"[Master    ] Saving data in run: {_run}")
        _dict = {
            'Actor': self.actor.state,
            'Optimizer_actor': self.ddpg.optimizer_actors.state_dict(),
            'Critic': self.critic.state,
            'Optimizer_critic': self.ddpg.optimizer_critics.state_dict(),
            'ReplayBuffer': self.replaybuffer
        }
        torch.save(_dict, _path)
        print("[Master    ] Data saved")

    def load(self, _run):
        if not _run:
            return
        print(f"[Master    ] Loading data from run: {_run}")
        _path = os.path.join(os.getcwd(), f'runs/{_run}/data.tar')
        if os.path.isfile(_path):
            _dict = torch.load(_path)
            self.critic.cnn_layers.load_state_dict(_dict['Encoder'])
            self.actor.cnn_layers.load_state_dict(_dict['Encoder'])
            self.critic.cnn_layers.eval()
            self.actor.cnn_layers.eval()
            print("[Master    ] Data loaded")
            return
        else:
            raise OSError('file not found')

    def test(self, t_step, n_tests=100):
        # Init environment
        rewards = []
        init_states_img = []
        predictions_img = []
        targets_img = []

        self.actor.set_eval()

        if self.demo:
            csv_actions = [[] for _ in range(self.n_token)]

        for _ in range(n_tests):
            self.env.create_random_setup()
            targets_img.append(self.env.target_img.squeeze())
            init_states_img.append(self.env.state_img.squeeze())


            for i in range(self.n_transitions):
                # Extract current state from environment
                state = self.env.state

                # Perform simulation step (performed on CPU)
                actions = self.actor(state, greedy=True).to("cpu")[self.env.control_nodes]

                for j, token in enumerate(self.env.tokens):
                    action = actions[j]
                    token.move(rel_vx=action[0], rel_vy=action[1], rel_wz=action[2])

                    if i == self.n_transitions - 1:
                        token.fix_position()

            predictions_img.append(self.env.state_img.squeeze())
            rewards.append(reward(self.env.state_img, self.env.target_img))

            if self.demo:
                for k in range(self.n_token):
                    token = self.env.tokens[k]
                    csv_actions[k].append((k, *token.center, token.orientation))

        print(f"[Master    ] Reached an average test score of {np.mean(rewards)}")
        self.writer.add_scalar(f'Test Results', np.mean(rewards), t_step)

        n_save = 3  # save top 3 results
        for i in range(n_save):
            n_best_predictions = np.argsort(rewards)[::-1][:n_save]
            self.save_result(np.array(init_states_img)[n_best_predictions],
                             np.array(predictions_img)[n_best_predictions],
                             np.array(targets_img)[n_best_predictions],
                             t_step)

        if self.demo:
            with open('demo.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerows(zip(*csv_actions))
