# -*- coding: utf-8 -*-
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from baselines.DDPG import Agent
from baselines.base import ReplayBuffer, NoiseGenerator


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        for layer in (0, 2, 4):
            nn.init.xavier_normal_(self.fc[layer].weight)
            nn.init.constant_(self.fc[layer].bias, 0)

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True)
        )
        self.fca = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        nn.init.xavier_normal_(self.fcs[0].weight)
        nn.init.constant_(self.fcs[0].bias, 0)
        nn.init.xavier_normal_(self.fca[0].weight)
        nn.init.constant_(self.fca[0].bias, 0)
        nn.init.xavier_normal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, 0)
        nn.init.xavier_normal_(self.fc[2].weight)
        nn.init.constant_(self.fc[2].bias, 0)

    def forward(self, state, action):
        return self.fc(torch.cat([self.fcs(state), self.fca(action)], 1))


# Fine-tune based on https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(NoiseGenerator):
    def __init__(self, size, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = None

    def generate(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
actor = Actor(state_dim, action_dim)
target_actor = copy.deepcopy(actor)
critic = Critic(state_dim, action_dim)
target_critic = copy.deepcopy(critic)
replay_module = ReplayBuffer(1e6)
noise_generator = OrnsteinUhlenbeckActionNoise(action_dim, 0, 0.2)
agent = Agent(actor, target_actor, Adam(actor.parameters(), 1e-4),
              critic, target_critic, Adam(critic.parameters(), 1e-3),
              nn.MSELoss(), replay_module, noise_generator)
agent.learn(env, 100000, 128)
