"""Replay Buffer."""

import random

from collections import namedtuple

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(experience)
            self.idx += 1
        else:
            self.idx = self.idx + 1 if self.idx < len(self.memory)-1 else 0
            self.memory[self.idx] = experience
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        memory_idxs = list(range(0, len(self.memory)-1))
        batch_idxs = random.sample(memory_idxs, batch_size)
        return [self.memory[idx] for idx in batch_idxs]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)