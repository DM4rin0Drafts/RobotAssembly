import torch as tr

gamma = 0.99


class Transition(object):

    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state

        self.reward = reward
        self.next_transition = None

    @property
    def terminal(self):
        if self.next_transition:
            return tr.tensor(0, dtype=tr.float32).to(self.state.device)
        else:
            return tr.tensor(1, dtype=tr.float32).to(self.state.device)

    @property
    def data(self):
        return [self.state, self.action, self.next_state, self.terminal, self.reward]
