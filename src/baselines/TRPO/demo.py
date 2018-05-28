# -*- coding: utf-8 -*-
import gym
import torch
import torch.nn as nn

from baselines import TRPO


class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )
        for layer in (0, 2):
            nn.init.xavier_normal_(self.fc[layer].weight)
            nn.init.constant_(self.fc[layer].bias, 0)

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, action_dim),
            nn.Softmax(1)
        )
        for layer in (0, 2):
            nn.init.xavier_normal_(self.fc[layer].weight)
            nn.init.constant_(self.fc[layer].bias, 0)

    def forward(self, x):
        return self.fc(x)


env = gym.make('CartPole-v0')
policy = Policy(env.observation_space.shape[0], env.action_space.n)
value = Value(env.observation_space.shape[0])
agent = TRPO.Agent(policy, value, nn.MSELoss(), torch.optim.Adam(value.parameters(), lr=1e-2))
agent.learn(env, 20000, 32)
