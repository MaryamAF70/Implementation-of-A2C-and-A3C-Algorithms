# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        # shared body
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # actor: output mean
        self.mu = nn.Linear(hidden, action_dim)
        # log std as parameter (state-independent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # critic: value
        self.v = nn.Linear(hidden, 1)

        # weight init
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        v = self.v(x).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std, v

    def get_action(self, obs):
        mu, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp, v, dist

    def get_logprob_and_value(self, obs, action):
        mu, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logp, entropy, v
