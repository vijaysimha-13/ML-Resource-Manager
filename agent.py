import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Actor
        self.actor_mean = nn.Linear(64, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        # state: (batch_size, state_dim)
        x = self.shared(state)
        mean = torch.sigmoid(self.actor_mean(x))     # [0,1]
        std = torch.exp(self.actor_log_std)           # learned std
        value = self.critic(x)
        return mean, std, value


class Agent:
    def __init__(self):
        self.state_dim = None
        self.model = None
        self.optimizer = None

        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.device = torch.device("cpu")

    def learn(self, env):
        # -------- Infer state dimension dynamically --------
        init_state = np.asarray(env.reset(), dtype=np.float32)
        self.state_dim = init_state.shape[0]

        # -------- Initialize model and optimizer --------
        if self.model is None:
            self.model = ActorCritic(self.state_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        max_episodes = 300
        max_steps = 300

        for episode in range(max_episodes):
            state = np.asarray(env.reset(), dtype=np.float32)
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)

            for _ in range(max_steps):
                mean, std, value = self.model(state_t)

                # Adaptive exploration
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().clamp(0.0, 1.0)

                action_np = action.detach().cpu().numpy().reshape(1)
                next_state, reward, done, _ = env.step(action_np)

                next_state = np.asarray(next_state, dtype=np.float32)
                next_state_t = (
                    torch.from_numpy(next_state)
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    _, _, next_value = self.model(next_state_t)

                reward_t = torch.tensor([[reward]], dtype=torch.float32)
                reward_t = torch.clamp(reward_t, -10.0, 10.0)

                target = reward_t + self.gamma * next_value * (1 - int(done))
                advantage = target - value

                actor_loss = -dist.log_prob(action) * advantage.detach()
                critic_loss = advantage.pow(2)
                entropy_loss = -dist.entropy()

                loss = actor_loss + critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                state_t = next_state_t

                if done:
                    break

    def get_action(self, state):
        if self.model is None:
            return np.array([np.random.rand()])

        state = np.asarray(state, dtype=np.float32)
        state_t = torch.from_numpy(state).unsqueeze(0)

        with torch.no_grad():
            mean, _, _ = self.model(state_t)

        return np.array([float(mean.item())])
