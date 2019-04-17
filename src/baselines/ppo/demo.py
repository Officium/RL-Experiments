# -*- coding: utf-8 -*-
import gym
import torch
import torch.nn as nn

from baselines import PPO


class AC(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Sequential(
            nn.Linear(10, action_dim),
            nn.LogSoftmax(1)
        )
        self.value = nn.Sequential(
            nn.Linear(10, 1)
        )
        for attr in (self.fc, self.policy, self.value):
            nn.init.xavier_normal_(attr[0].weight)
            nn.init.constant_(attr[0].bias, 0)

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)


env = gym.make('CartPole-v0')
ac = AC(env.observation_space.shape[0], env.action_space.n)
agent = PPO.Agent(ac, nn.MSELoss(),
                  torch.optim.Adam(ac.parameters(), lr=1e-3))
agent.learn(env, 200)
