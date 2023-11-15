import torch

from environment.tangram import Tangram
from models.actor_mha import Actor
from simulation.trajectory import Trajectory
from simulation.transition import Transition
from utilities.reward import reward

class Worker(object):

    def __init__(self, args, worker_id, device="cpu"):
        self.worker_id = worker_id
        self.device = device
        self.n_iterations = args.n_iterations
        self.n_transitions = args.n_transitions

        self.env = Tangram(tokens=args.tokens, gui=args.gui, device=device)
        self.env.create_random_setup()

        self.actor = Actor(n_features=args.n_features_actor,
                           n_layers=args.n_layers,
                           embed_dim=args.embed_dim,
                           n_heads=args.n_heads,
                           n_hidden_readout=args.n_hidden_readout,
                           dim_single_pred=args.dim_single_pred_actor,
                           device=device)

        # TODO: Find solution for parameterization of Gaussian for token spawn sampling
        self.__mean = 1
        self.__lmbda = 1

    def simulate_rollout(self, parameters):
        self.actor.state = parameters
        self.actor.eval()  # turn off train mode

        trajectories = []
        while len(trajectories) < self.n_iterations:
            self.env.create_random_setup()  # creates target
            original_trajectory = Trajectory()
            state_imgs = []

            for i in range(self.n_transitions):
                state = self.env.state

                actions = self.actor(state, greedy=True)[self.env.control_nodes]

                for j, token in enumerate(self.env.tokens):
                    action = actions[j]
                    token.move(rel_vx=action[0], rel_vy=action[1], rel_wz=action[2])

                    if i == self.n_transitions - 1:
                        token.fix_position()

                next_state = self.env.state

                original_reward = reward(self.env.state_img, self.env.target_img)
                state_imgs.append(self.env.state_img)

                original_trajectory.add_transition(
                        Transition(state, actions, next_state, original_reward))

            trajectories.append(original_trajectory)

            hindsight_trajectory = Trajectory()
            terminal_state = original_trajectory.transitions[-1].next_state
            terminal_state_img = state_imgs[-1]
            for i, transition in enumerate(original_trajectory.transitions):
                # overwrite state and next state (because change of target)
                hindsight_state = self.env.overwrite_target_in_state(transition.state, terminal_state)
                hindsight_action = transition.action
                hindsight_next_state = self.env.overwrite_target_in_state(transition.next_state, terminal_state)
                hindsight_reward = reward(state_imgs[i], terminal_state_img)

                hindsight_trajectory.add_transition(
                    Transition(hindsight_state, hindsight_action, hindsight_next_state, hindsight_reward))

            trajectories.append(hindsight_trajectory)

        return self.__trajectories_as_list_per_token(trajectories)

    def __trajectories_as_list_per_token(self, trajectories):
        #trajectories_per_token = [[] for _ in range(self.env.n_token)]
        trajectories_per_token = []
        for trajectory in trajectories:
            for transition in trajectory.transitions:
                trajectories_per_token.append(transition.data)
        return trajectories_per_token
