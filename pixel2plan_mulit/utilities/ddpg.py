""" Deep deterministic Policy Gradient
DDPG paper: Lillicrap et. al, 2015, Continuous control with deep reinforcement learning
DDPG inspired by Sameera Lanka, https://github.com/samlanka/DDPG-Pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DDPG(object):

    def __init__(self, actor, critic, lr_actor=1e-4, lr_critic=1e-3, gamma=0.9, tau=1e-3):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau

        assert actor.device == critic.device

        # Initialize critic
        self.critic = critic
        self.optimizer_critics = optim.Adam(self.critic.parameters(), self.lr_critic)

        # Initialize actors
        self.actor = actor
        self.optimizer_actors = optim.Adam(self.actor.parameters(), self.lr_actor)

        self.loss_critic = nn.MSELoss().to(self.device)

    @property
    def n_critics(self):
        return self.critic.n_heads

    @property
    def n_actors(self):
        return self.actor.n_heads

    @property
    def device(self):
        return self.actor.device

    def bootstrapping(self, next_actor_batch, next_s_batch, d_batch, r_batch, t_batch):
        """ Inputs: Batch of next states, rewards and targets of size self.batchSize
            Calculates the target Q-value from reward and bootstraped Q-value of next state
            using the target actor and target critic
           Outputs: Batch of Q-value targets """
        for i in range(self.n_actors):
            indices = np.argwhere(next_actor_batch == i).squeeze(0)
            if indices.numel() > 0:
                with torch.no_grad():
                    next_action_batch = self.target_actor(
                        next_s_batch[indices], t_batch[indices], i, greedy=True
                    ).detach()
                next_q_batch = self.target_critic(
                    next_s_batch[indices], t_batch[indices], next_action_batch, i, greedy=True
                )

                r_batch[indices] = r_batch[indices] + self.gamma * next_q_batch.data.T.squeeze(0)
        return r_batch

    def update_targets(self, target, original):
        """ Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic) """
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - self.tau) * targetParam.data + self.tau * orgParam.data)

    def critic_step(self, index, batch, writer=None, t_step=0):
        # Batch is given as [states, actions, next states, terminal flags, rewards/values]
        # where each component is a vector
        s_batch, a_batch, next_s_batch, term_batch, v_batch = batch


        q_value_batch = self.critic(s_batch, a_batch)
        # q_target_batch = self.bootstrapping(next_actor_batch, next_s_batch, d_batch, r_batch, t_batch).unsqueeze(1)
        q_target_batch = v_batch.unsqueeze(1)

        # Critic update
        self.optimizer_critics.zero_grad()
        critic_loss = self.loss_critic(q_value_batch, q_target_batch)
        critic_loss.backward()
        self.optimizer_critics.step()

        if writer:
            print(f"[DDPG      ] Critic Loss: {critic_loss:.2f}")
            writer.add_scalar(f'Mean batch target Q-Value {index}', torch.mean(q_target_batch), t_step)
            writer.add_scalar(f'Mean batch predicted Q-Value {index}', torch.mean(q_value_batch), t_step)
            writer.add_scalar(f'Critic Loss {index}', critic_loss.item(), t_step)
        return

    def actor_step(self, index, batch, writer=None, t_step=0):
        # Batch is given as [states, actions, next states, terminal flags, rewards/values]
        # where each component is a vector
        s_batch, a_batch, _, _, _ = batch  # TODO: t_batch (targets) is included in s_batch (states)

        # Actor update
        self.optimizer_actors.zero_grad()
        predicted_a_batch = self.actor(s_batch, greedy=False)[:, self.critic.control_nodes, :]
        actor_loss = -torch.mean(self.critic(s_batch, predicted_a_batch))
        actor_loss.backward()
        self.optimizer_actors.step()

        if writer:
            print(f"[DDPG      ] Actor Loss: {actor_loss:.2f}")
            writer.add_scalar(f'actor_loss_{index}', actor_loss.item(), t_step)
            # TODO: attribute "self.actor.log_std" not defined
            #writer.add_scalar(f'Std Actor {index} in x', torch.exp(self.actor.log_std[index][0, 0]).item(), t_step)
            #writer.add_scalar(f'Std Actor {index} in y', torch.exp(self.actor.log_std[index][0, 1]).item(), t_step)
            #writer.add_scalar(f'Std Actor {index} in phi', torch.exp(self.actor.log_std[index][0, 2]).item(), t_step)
        return

    def train(self, index, buffer, batch_size, writer=None, t_step=0):
        self.critic.set_train()

        # n+1 critic updates
        self.critic_step(index, buffer.sample(batch_size), writer=writer, t_step=t_step)
        for n in range(49):
            self.critic_step(index, buffer.sample(batch_size))

        self.actor.set_train()
        self.critic.set_eval()

        # m+1 actor updates
        self.actor_step(index, buffer.sample(batch_size), writer=writer, t_step=t_step)
        for m in range(0):
            self.actor_step(index, buffer.sample(batch_size))
        return
