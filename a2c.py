#a2c
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time


# -------------------------
# Actor-Critic Network
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Actor: outputs mean of Gaussian distribution
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

        # Std for Gaussian policy (trainable parameter)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        value = self.critic(x)
        return mean, std, value


# -------------------------
# A2C Training Function
# -------------------------
def train_a2c(args):
    env = gym.make("Pendulum-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    model = ActorCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    all_rewards = []
    start_time = time.time()

    for episode in range(args.max_episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
            mean, std, value = model(state_t)

            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_clipped = torch.clamp(action, -action_bound, action_bound)

            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy()[0])
            done = terminated or truncated

            reward = float(reward)  # fix: numpy.ndarray → float

            rewards.append(torch.tensor([reward], dtype=torch.float32))
            values.append(value)
            log_probs.append(dist.log_prob(action).sum(dim=-1, keepdim=True))

            state = next_state

        # Compute returns
        returns = []
        G = torch.zeros(1)
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)

        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        advantage = returns - values.squeeze()

        # Actor loss (policy gradient)
        actor_loss = -(log_probs.squeeze() * advantage.detach()).mean()
        # Critic loss (MSE)
        critic_loss = advantage.pow(2).mean()
        # Total loss
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum([r.item() for r in rewards])
        all_rewards.append(episode_reward)

        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(all_rewards[-args.log_interval:])
            print(f"Episode {episode+1}, Avg Reward (last {args.log_interval} eps): {avg_reward:.2f}")

    elapsed_time = time.time() - start_time
    env.close()
    return all_rewards, elapsed_time