# -*- coding: utf-8 -*-
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import PPO


class AC(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AC, self).__init__()
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2a = nn.Linear(10, action_dim)
        self.fc2a.weight.data.normal_(0, 0.1)
        self.softmax = nn.Softmax(1)
        self.fc2c = nn.Linear(10, 1)
        self.fc2c.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.softmax(self.fc2a(x)), self.fc2c(x)


env = gym.make('CartPole-v0')
ac = AC(env.observation_space.shape[0], env.action_space.n)
agent = PPO.Agent(ac, nn.MSELoss(),
                  torch.optim.Adam(ac.parameters(), lr=1e-3))
agent.learn(env, 200)
