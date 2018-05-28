# -*- coding: utf-8 -*-
import copy

import gym
import torch
import torch.nn as nn

from baselines import DQN, base


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim)
        )
        for layer in (0, 2):
            nn.init.xavier_normal_(self.fc[layer].weight)
            nn.init.constant_(self.fc[layer].bias, 0)

    def forward(self, x):
        return self.fc(x)


env = gym.make('CartPole-v0')
q = Net(env.observation_space.shape[0], env.action_space.n)
target_q = copy.deepcopy(q)
agent = DQN.Agent(q, target_q, base.ReplayBuffer(50000),
                  torch.optim.Adam(q.parameters(), lr=1e-3), nn.MSELoss(), 500, reward_gamma=1.0)
agent.learn(env, 1000, 32)
