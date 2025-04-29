import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            np.vstack(states),
            np.vstack(actions),
            np.array(rewards).reshape(-1, 1),
            np.vstack(next_states),
            np.array(dones).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)
