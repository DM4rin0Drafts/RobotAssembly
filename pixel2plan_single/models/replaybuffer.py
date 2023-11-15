from collections import deque
import torch as tr
import random


class ReplayBuffer(object):

    def __init__(self, buffer_size, device="cpu"):
        self.count = 0
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.device = device

    def add(self, trajectory):
        for transition in trajectory:
            if self.count < self.buffer_size:
                self.buffer.append(transition)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(transition)

    def sample(self, batch_size):
        """ Sample batch of batch_size. If less samples are stored return all. """
        # Check current number of stored transitions
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch, a_batch, next_s_batch, term_batch, v_batch = list(map(tr.stack, list(zip(*batch))))

        return s_batch.to(self.device), a_batch.to(self.device), next_s_batch.to(self.device), \
               term_batch.to(self.device), v_batch.to(self.device)

    def clear(self):
        self.buffer.clear()
        self.count = 0
