import numpy as np

class DummyEnv:
    def reset(self):
        return np.array([0.5, 0.5])

    def step(self, action):
        reward = 1 - abs(action[0] - 0.5)
        return np.array([0.5, 0.5]), reward, True, {}
