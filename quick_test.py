# quick_test.py
from env import Environment
import numpy as np

env = Environment()
state = env.reset()

for i in range(10):
    action = np.array([0.5])
    state, reward, done, _ = env.step(action)
    print(f"Step {i}: state={state}, reward={reward:.3f}")
    if done:
        break
