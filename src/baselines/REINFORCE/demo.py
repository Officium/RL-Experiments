# -*- coding: utf-8 -*-
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import REINFORCE


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, action_dim)
        self.out.weight.data.normal_(0, 0.1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_prob = self.softmax(self.out(x))
        return actions_prob


env = gym.make('CartPole-v0')
policy = Policy(env.observation_space.shape[0], env.action_space.n)
agent = REINFORCE.Agent(policy, torch.optim.Adam(policy.parameters(), lr=1e-2))
agent.learn(env, 200, 20)
