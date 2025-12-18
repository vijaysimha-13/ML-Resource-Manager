import numpy as np

class Environment:
    def __init__(self):
        self.S_low_limits = np.array([0.0, 0.0])
        self.S_upper_limits = np.array([1.0, 1.0])

        self.A_low_limits = np.array([0.0])
        self.A_upper_limits = np.array([1.0])

        self.max_steps = 200
        self.t = 0

        self.resource = None
        self.health = None

        self.regime_timer = None
        self.regime_length_range = (20, 50)

    def reset(self):
        self.resource = 1.0
        self.health = 0.5
        self.t = 0

        self.regime_timer = np.random.randint(
            self.regime_length_range[0],
            self.regime_length_range[1]
        )

        return self._get_state()

    def step(self, action):
        action = np.clip(action[0], 0.0, 1.0)
        self.t += 1
        self.regime_timer -= 1

        # Resource dynamics
        base_cost = 0.04
        invest_cost = 0.25 * action
        noise = np.random.normal(0, 0.01)

        self.resource = np.clip(
            self.resource - base_cost - invest_cost + noise,
            0.0,
            1.0
        )

        # Health dynamics (non-stationary)
        if self.regime_timer > 0:
            health_delta = 0.10 * action - 0.02
        else:
            health_delta = 0.10 * (1 - action) - 0.02

        self.health = np.clip(self.health + health_delta, 0.0, 1.0)

        if self.regime_timer <= 0:
            self.regime_timer = np.random.randint(
                self.regime_length_range[0],
                self.regime_length_range[1]
            )

        # ---------------- Reward (numerically stable) ----------------
        # Immediate reward (small, bounded)
        reward = 0.1 * (1 - action)

        # Smooth delayed reward (bounded)
        if self.resource > 0.2:
            reward += 0.5 * max(0.0, self.health - 0.8)

        # Ensure finite reward
        reward = float(np.clip(reward, -1.0, 1.0))

        # Termination
        done = self.resource <= 0.0 or self.t >= self.max_steps

        info = {}

        return self._get_state(), reward, done, info

    def _get_state(self):
        return np.array([self.resource, self.health])
