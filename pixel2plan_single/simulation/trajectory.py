class Trajectory(object):

    def __init__(self):
        self.transitions = None

    @property
    def length(self):
        if self.transitions:
            return len(self.transitions)
        else:
            return 0

    def add_transition(self, transition):
        if self.length > 0:
            self.transitions[-1].next_transition = transition
            self.transitions.append(transition)
        else:
            self.transitions = [transition]
