# -*- coding: utf-8 -*-
import math
import copy

import numpy
import torch
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    def __init__(self, policy, value, accept_ratio, reward_gamma=0.9):
        self._policy = policy
        self._value = value
        self.accept_ratio = accept_ratio
        self.reward_gamma = reward_gamma

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0), requires_grad=True)
        prob = self._policy(state)
        action = prob.multinomial(1)
        return action, prob

    def sample_trajectories(self, env, episodes):
        trajectories = []
        i_episodes = 0
        entropy = 0
        rewards = 0
        while i_episodes < episodes:
            i_episodes += 1
            s = env.reset()
            data = []
            while True:
                a, p = self.act(s)
                entropy += -(p * p.log()).sum()
                s_, r, done, info = env.step(a.data[0, 0])
                rewards += r
                data.append(tuple((s, a, r, p)))
                if done:
                    trajectories.append([None] * len(data))
                    trajectories[-1][-1] = data[-1]
                    for i in reversed(xrange(len(data)-1)):
                        trajectories[-1][i] = tuple((data[i][0], data[i][1],
                                                     data[i][2] + self.reward_gamma * data[i+1][2], data[i][3]))
                    break
                s = s_
        rewards /= episodes
        entropy /= len(trajectories)
        return trajectories, rewards, entropy

    def loss(self, state_dict, b_s, b_a, advantage):
        prob_old = self._policy(b_s).gather(1, b_a).data
        new_model = copy.deepcopy(self._policy)
        new_model.load_state_dict(state_dict)
        prob_new = new_model(b_s).gather(1, b_a).data
        return -torch.mean((prob_new / (prob_old + 1e-8)) * advantage)

    def learn(self, env, max_iter, batch_size, sample_episodes):
        for i_episode in xrange(max_iter):
            trajectories, rewards, entropy = self.sample_trajectories(env, sample_episodes)
            for trajectory in numpy.array_split(trajectories, math.ceil(len(trajectories) * 1.0 / batch_size)):
                b_s = Variable(torch.Tensor([x[0] for x in trajectory]))
                b_r = torch.Tensor([[x[1]] for x in trajectory])
                b_a = torch.Tensor([x[2] for x in trajectory])
                b_p = torch.Tensor([x[3] for x in trajectory])

                baseline = self._value.forward(b_s).data
                advantage = b_r - baseline
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
