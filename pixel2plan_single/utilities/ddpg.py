""" Deep deterministic Policy Gradient
DDPG paper: Lillicrap et. al, 2015, Continuous control with deep reinforcement learning
DDPG inspired by Sameera Lanka, https://github.com/samlanka/DDPG-Pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim

from models.actor_mha import ActorMHA
from models.critic_mha import CriticMHA


class DDPG:
    def __init__(self, actor, actor_target, critic, critic_target, lr_actor=1e-4,
                 lr_critic=1e-3, gamma=0.9, tau=1e-3):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau

        assert actor.device == critic.device
        self.device = actor.device

        self.critic = critic
        self.critic_target = critic_target
        self.optimizer_critics = optim.Adam(self.critic.parameters(), self.lr_critic)
        self.loss_critic = nn.MSELoss().to(self.device)

        self.actor = actor
        self.actor_target = actor_target
        self.optimizer_actors = optim.Adam(self.actor.parameters(), self.lr_actor)

        self.update_target_parameters(polyak_averaging=False)

    def update_target_parameters(self, polyak_averaging=True):
        """
        Updates target network (critic and actor) parameters based on corresponding main networks.

        Args:
            polyak_averaging: If true, performs polyak averaging for network update.
                              Else, performs exact copy of network parameters.
        """
        tau = 1 if not polyak_averaging else self.tau
        actor_weights = self.actor.state_dict().copy()
        critic_weights = self.critic.state_dict().copy()
        target_actor_weights = self.actor_target.state_dict().copy()
        target_critic_weights = self.critic_target.state_dict().copy()

        assert actor_weights.keys() == critic_weights.keys()
        for key in actor_weights.keys():
            target_actor_weights[key] = (1 - tau) * target_actor_weights[key] + tau * actor_weights[key]
            target_critic_weights[key] = (1 - tau) * target_critic_weights[key] + tau * critic_weights[key]

        self.actor_target.load_state_dict(target_actor_weights)
        self.critic_target.load_state_dict(target_critic_weights)

    def critic_step(self, batch, writer=None, t_step=0):
        # Batch is given as [states, actions, next states, terminal flags, rewards/values]
        # where each component is a vector
        states, actions, next_states, is_terminals, rewards = batch

        q_values = self.critic(states, actions)
        target_actions = self.actor_target(next_states, greedy=True)
        q_target_values = rewards + self.gamma * self.critic_target(next_states, target_actions) * (1 - is_terminals)

        # Critic update
        self.optimizer_critics.zero_grad()
        critic_loss = self.loss_critic(q_values, q_target_values)
        critic_loss.backward()
        self.optimizer_critics.step()

        if writer:
            print(f"[DDPG      ] Critic Loss: {critic_loss:.2f}")
            writer.add_scalar(f'Mean batch target Q-Value', torch.mean(q_target_values), t_step)
            writer.add_scalar(f'Mean batch predicted Q-Value', torch.mean(q_values), t_step)
            writer.add_scalar(f'Critic Loss', critic_loss.item(), t_step)
        return

    def actor_step(self, batch, writer=None, t_step=0):
        # Batch is given as [states, actions, next states, terminal flags, rewards/values]
        # where each component is a vector
        states, actions, _, _, _ = batch

        # Actor update
        self.optimizer_actors.zero_grad()
        predicted_actions = self.actor(states, greedy=True)
        actor_loss = -torch.mean(self.critic(states, predicted_actions))  # negative because we apply gradient ascent
        actor_loss.backward()
        self.optimizer_actors.step()

        if writer:
            print(f"[DDPG      ] Actor Loss: {actor_loss:.2f}")
            writer.add_scalar(f'actor_loss', actor_loss.item(), t_step)
        return

    def train(self, buffer, batch_size, writer=None, t_step=0):
        self.critic.train()
        self.critic_step(buffer.sample(batch_size), writer=writer, t_step=t_step)
        for _ in range(50):
            self.critic_step(buffer.sample(batch_size))
        self.critic.eval()

        self.actor.train()
        self.actor_step(buffer.sample(batch_size), writer=writer, t_step=t_step)
        for _ in range(0):
            self.actor_step(buffer.sample(batch_size))
        self.actor.eval()

        self.update_target_parameters()
