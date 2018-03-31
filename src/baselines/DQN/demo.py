# -*- coding: utf-8 -*-
import copy

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import DQN, base


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, action_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


env = gym.make('CartPole-v0')
q = Net(env.observation_space.shape[0], env.action_space.n)
target_q = copy.deepcopy(q)
agent = DQN.Agent(q, target_q, base.ReplayBuffer(1000),
                  torch.optim.Adam(q.parameters(), lr=1e-2), nn.MSELoss(), 100)
agent.learn(env, 1000, 100)
