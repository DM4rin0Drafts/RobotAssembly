from typing import List

from constants import actor_structure
from environment import Token
from environment.tangram import Tangram
from models.actor_fc import ActorFC
from models.actor_mha import ActorMHA
from simulation.training_setup import TrainingSetup, Architecture, Reward
from simulation.trajectory import Trajectory
from simulation.transition import Transition
from utilities.reward import sparse_reward, dense_reward


class Worker:
    def __init__(self, worker_id, tokens: List[Token], train_setup: TrainingSetup, gui=False, device="cpu"):
        self.worker_id = worker_id
        self.device = device
        self.n_iterations = train_setup.n_iterations
        self.reward_func = train_setup.reward

        self.overfit = train_setup.overfit
        self.env = Tangram(tokens, gui, constant_setup_only=self.overfit, device=self.device)
        if self.overfit:
            self.env.create_constant_setup()
        else:
            self.env.create_random_setup()

        if train_setup.architecture == Architecture.MHA:
            self.actor = ActorMHA(actor_structure.n_features,
                                  actor_structure.n_layers,
                                  actor_structure.embed_dim,
                                  actor_structure.n_heads,
                                  actor_structure.n_hidden_readout,
                                  actor_structure.dim_single_pred,
                                  train_setup.log_std,
                                  self.device)

        elif train_setup.architecture == Architecture.FC_Small:
            self.actor = ActorFC(actor_structure.n_features, actor_structure.dim_single_pred, train_setup.log_std,
                                 small_fc=True, device=self.device)

        elif train_setup.architecture == Architecture.FC_Large:
            self.actor = ActorFC(actor_structure.n_features, actor_structure.dim_single_pred, train_setup.log_std,
                                 small_fc=False, device=self.device)

    def simulate_rollout(self, parameters):
        self.actor.state = parameters
        self.actor.eval()  # turn off train mode

        trajectories = []
        while len(trajectories) < self.n_iterations:
            token_idx = 0  # TODO: Generalize index to multiple tokens

            if self.overfit:
                self.env.create_constant_setup()
            else:
                self.env.create_random_setup()  # creates target
            original_trajectory = Trajectory()

            state = self.env.state
            action = self.actor(self.env.state, greedy=False).squeeze()
            self.env.tokens[token_idx].move(rel_vx=action[0], rel_vy=action[1], rel_wz=action[2])

            next_state = self.env.state

            if self.reward_func == Reward.SPARSE:
                original_reward = sparse_reward(self.env.state_img, self.env.target_img)
            else:
                original_reward = dense_reward(next_state)

            original_trajectory.add_transition(
                Transition(state, action, next_state, original_reward))
            trajectories.append(original_trajectory)

            hindsight_trajectory = Trajectory()
            hindsight_state = self.env.overwrite_target_in_state(state, next_state)
            hindsight_next_state = self.env.overwrite_target_in_state(next_state, next_state)

            if self.reward_func == Reward.SPARSE:
                hindsight_reward = sparse_reward(self.env.state_img, self.env.state_img)
            else:
                hindsight_reward = dense_reward(hindsight_next_state)

            hindsight_trajectory.add_transition(
                Transition(hindsight_state, action, next_state, hindsight_reward))
            trajectories.append(hindsight_trajectory)
        return self.__trajectories_as_list_per_token(trajectories)

    def __trajectories_as_list_per_token(self, trajectories):
        trajectories_per_token = []
        for trajectory in trajectories:
            for transition in trajectory.transitions:
                trajectories_per_token.append(transition.data)
        return trajectories_per_token
