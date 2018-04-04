# -*- coding: utf-8 -*-
import copy

import numpy as np

import gym
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from baselines.DDPG import Agent
from baselines.base import ReplayBuffer, NoiseGenerator


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        t = 1.0 / np.sqrt(128)
        self.fc1.weight.data.uniform_(-t, t)

        self.fc2 = nn.Linear(128, 128)
        t = 1.0 / np.sqrt(128)
        self.fc2.weight.data.uniform_(-t, t)

        self.fc3 = nn.Linear(128, action_dim)
        self.fc3.weight.data.uniform_(init_w, init_w)

    def forward(self, x):
        return F.tanh(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1s = nn.Linear(state_dim, 128)
        t = 1.0 / np.sqrt(128)
        self.fc1s.weight.data.uniform_(-t, t)

        self.fc1a = nn.Linear(action_dim, 128)
        t = 1.0 / np.sqrt(128)
        self.fc1a.weight.data.uniform_(-t, t)

        self.fc2 = nn.Linear(128 + 128, 128)
        t = 1.0 / np.sqrt(128)
        self.fc2.weight.data.uniform_(-t, t)

        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(init_w, init_w)

    def forward(self, state, action):
        return self.fc3(F.relu(self.fc2(torch.cat([F.relu(self.fc1s(state)), F.relu(self.fc1a(action))], 1))))


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
