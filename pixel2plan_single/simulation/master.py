import os
import csv
from typing import List

import matplotlib.pyplot as plt
from numpy import ndarray

from torch.utils.tensorboard import SummaryWriter

from constants import actor_structure, critic_structure
from environment import Token
from models.actor_fc import ActorFC, actor_fc_from
from models.critic_fc import CriticFC, critic_fc_from
from models.replaybuffer import ReplayBuffer
from models.critic_mha import CriticMHA, critic_mha_from
from models.actor_mha import ActorMHA, actor_mha_from
from environment.tangram import Tangram
from simulation.training_setup import TrainingSetup, Architecture, Reward
from utilities.ddpg import DDPG
from utilities.reward import sparse_reward, dense_reward
from utilities.utilities import *


class Master:
    def __init__(self, tokens: List[Token], train_setup: TrainingSetup, run: str or None, gui=False, demo=False,
                 device="cpu"):
        self.demo = demo
        self.device = device
        self.overfit = train_setup.overfit
        self.reward_func = train_setup.reward
        self.compare_rewards = train_setup.compare_rewards

        self.n_episodes = train_setup.n_episodes

        self.env = Tangram(tokens, gui, constant_setup_only=self.overfit, device=self.device)

        # TODO: Actor and Critic from in utilities
        if train_setup.architecture == Architecture.MHA:
            architecture = "MHA"
            self.actor = ActorMHA(actor_structure.n_features,
                                  actor_structure.n_layers,
                                  actor_structure.embed_dim,
                                  actor_structure.n_heads,
                                  actor_structure.n_hidden_readout,
                                  actor_structure.dim_single_pred,
                                  train_setup.log_std,
                                  self.device)
            self.actor_target = actor_mha_from(self.actor)
            self.critic = CriticMHA(critic_structure.n_features,
                                    critic_structure.n_layers,
                                    critic_structure.embed_dim,
                                    critic_structure.n_heads,
                                    critic_structure.n_hidden_readout,
                                    critic_structure.dim_single_pred,
                                    train_setup.log_std,
                                    self.device)
            self.critic_target = critic_mha_from(self.critic)

        elif train_setup.architecture == Architecture.FC_Small:
            architecture = "FC_Small"
            self.actor = ActorFC(actor_structure.n_features, actor_structure.dim_single_pred, train_setup.log_std,
                                 small_fc=True, device=self.device)
            self.actor_target = actor_fc_from(self.actor)
            self.critic = CriticFC(critic_structure.n_features, critic_structure.dim_single_pred, train_setup.log_std,
                                   small_fc=True, device=self.device)
            self.critic_target = critic_fc_from(self.critic)

        elif train_setup.architecture == Architecture.FC_Large:
            architecture = "FC_Large"
            self.actor = ActorFC(actor_structure.n_features, actor_structure.dim_single_pred, train_setup.log_std,
                                 small_fc=False, device=self.device)
            self.actor_target = actor_fc_from(self.actor)
            self.critic = CriticFC(critic_structure.n_features, critic_structure.dim_single_pred, train_setup.log_std,
                                   small_fc=False, device=self.device)
            self.critic_target = critic_fc_from(self.critic)

        reward_func = "DENSE" if train_setup.reward == Reward.DENSE else "SPARSE"
        run_suffix = "_" + architecture + "_" + reward_func
        self.writer = SummaryWriter(comment=run_suffix)
        self.writer.add_graph(self.actor, torch.zeros(9, 13).to(self.device))

        self.replaybuffer = ReplayBuffer(train_setup.buffer_size, device)
        if not self.demo:
            self.ddpg = DDPG(self.actor, self.actor_target, self.critic, self.critic_target,
                             lr_actor=train_setup.lr_actor, lr_critic=train_setup.lr_critic,
                             gamma=train_setup.gamma, tau=train_setup.tau)

        if run:
            self.load(run)


        actor_params = sum(param.numel() for param in self.actor.parameters())
        critic_params = sum(param.numel() for param in self.critic.parameters())
        print(f"[Master    ] Total number of parameters (Actor + Critic): {actor_params + critic_params}")

        self.n_finished_jobs = 0
        self.n_given_jobs = 0

    @property
    def n_token(self):
        return self.env.n_token

    def iteration(self):
        pass

    def run(self):
        while self.n_finished_jobs < self.n_episodes:
            if self.n_finished_jobs != 0 and self.n_finished_jobs % 1000 == 0:
                self.save()
            self.iteration()

    def save(self, run: str or None = None, n_ckpt=0):
        if run is None:
            run = self.writer.get_logdir()
        _run = run

        _path = os.path.join(os.getcwd(), f'{_run}/data{n_ckpt}.tar')
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
        rewards_sparse = []
        rewards_dense = []
        init_states_img = []
        predictions_img = []
        targets_img = []
        actions = []

        self.actor.eval()

        if self.demo:
            csv_actions = [[] for _ in range(self.n_token)]

        for _ in range(n_tests):
            if self.overfit:
                self.env.create_constant_setup()
            else:
                self.env.create_random_setup()  # create random target and initial token setup
            token_idx = 0
            targets_img.append(self.env.target_img.squeeze())
            init_states_img.append(self.env.state_img.squeeze())

            # Extract current state from environment
            state = self.env.state

            # Perform simulation step (performed on CPU)
            action = self.actor(state, greedy=True).squeeze().to("cpu")
            self.env.tokens[token_idx].move(rel_vx=action[0], rel_vy=action[1], rel_wz=action[2])
            actions.append(action.detach().clone().numpy())

            predictions_img.append(self.env.state_img.squeeze())
            if self.compare_rewards:
                rewards_sparse.append(sparse_reward(self.env.state_img, self.env.target_img))
                rewards_dense.append(dense_reward(self.env.state))
            else:
                if self.reward_func == Reward.SPARSE:
                    rewards_sparse.append(sparse_reward(self.env.state_img, self.env.target_img))
                else:
                    rewards_dense.append(dense_reward(self.env.state))

            if self.demo:
                for k in range(self.n_token):
                    token = self.env.tokens[k]
                    csv_actions[k].append((k, *token.center, token.orientation))

        rewards = rewards_sparse if self.reward_func == Reward.SPARSE else rewards_dense

        if self.compare_rewards:
            print(f"[Master    ] Reached an average test score of {np.mean(rewards_sparse)} (Sparse)")
            print(f"[Master    ] Reached an average test score of {np.mean(rewards_dense)} (Dense)")
            self.writer.add_scalar(f'Test Results (Sparse)', np.mean(rewards_sparse), t_step)
            self.writer.add_scalar(f'Test Results (Dense)', np.mean(rewards_dense), t_step)
        else:
            print(f"[Master    ] Reached an average test score of {np.mean(rewards)}")
            self.writer.add_scalar(f'Test Results', np.mean(rewards), t_step)

        n_save = 3
        show_best = False
        reward_type = "dense"
        for i in range(n_save):
            if show_best:
                chosen_predictions = np.argsort(rewards)[::-1][:n_save]
            else:  # choose n random predictions
                chosen_predictions = np.random.choice(len(rewards), size=n_save)

            init_img = self.convertImgs(np.array(init_states_img)[chosen_predictions])
            pred_img = self.convertImgs(np.array(predictions_img)[chosen_predictions])
            target_imgs = self.convertImgs(np.array(targets_img)[chosen_predictions])
            a = np.array(actions)[chosen_predictions]
            r = np.array(rewards)[chosen_predictions]

            self.save_plots(init_img, pred_img, target_imgs, a, r, t_step, show_best)

        if self.demo:
            with open('demo.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerows(zip(*csv_actions))

    def load_target(self, _path):
        target_img_rgb = plt.imread(_path)
        target_img_binary = rgb2binary(target_img_rgb)
        self.env.target_img = target_img_rgb
        self.env.target = numpy2torch(target_img_binary)

    def convertImgs(self, imgs):
        for i, img in enumerate(imgs):
            tmp = img.clone().detach()

            nImg = np.zeros(tuple(tmp.shape), dtype=np.intc)
            nImg[(tmp==False).cpu().numpy()] = 239
            nImg[(tmp==True).cpu().numpy()] = 0
            imgs[i] = torch.as_tensor(nImg).to(self.device).int()

        return imgs


    def save_plots(self, init_states: ndarray, states_after_action: ndarray, target_states: ndarray, actions: ndarray,
                   rewards: ndarray, t_step: int, only_best: bool):
        fig = self.__create_plots(init_states, states_after_action, target_states, actions, rewards)
        if only_best:
            self.writer.add_figure(f"Episode {t_step} Results (Top {len(init_states)})", fig, t_step)
        else:
            self.writer.add_figure(f"Episode {t_step} Results", fig, t_step)

    @staticmethod
    def __create_plots(init_states: ndarray, states_after_action: ndarray, target_states: ndarray, actions: ndarray,
                       rewards: ndarray) -> plt.Figure:
        rows = len(init_states)
        fig, axs = plt.subplots(rows, 4)
        for i in range(rows):
            axs[i, 0].imshow(init_states[i].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            axs[i, 1].imshow(states_after_action[i].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            axs[i, 1].set_title("a = [{:.1f}, {:.1f}, {:.1f}]".format(actions[i][0], actions[i][1], actions[i][2]))
            axs[i, 2].imshow(target_states[i].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            axs[i, 3].imshow((states_after_action[i].cpu().numpy() + target_states[i].cpu().numpy()) / 2, cmap="gray", vmin=0, vmax=255)
            axs[i, 3].set_title("{:.2f}".format(rewards[i]))

            for j in range(len(axs[i])):
                axs[i, j].axis("off")

        axs[0, 0].set_title("Initial")
        axs[0, 2].set_title("Target")
        return fig
