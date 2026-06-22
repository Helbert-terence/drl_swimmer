import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dim)
        self.log_sigma = nn.Parameter(torch.zeros(action_dim))
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return self.max_action * x
    
    def act(self, state):
        mu = self.forward(state)
        sigma = self.log_sigma.clamp(-20, 2).exp()
        dist = Normal(mu, sigma)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, states, actions):
        mu = self.forward(states)
        sigma = self.log_sigma.clamp(-20, 2).exp()
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
    
class Critic(nn.Module):
    def __init__(self, state_dim,):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)