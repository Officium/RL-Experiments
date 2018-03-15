# -*- coding: utf-8 -*-
import copy

import numpy
import torch
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    def __init__(self, policy, value, accept_ratio=0.9, reward_gamma=0.9):
        self._policy = policy
        self._value = value
        self.accept_ratio = accept_ratio
        self.reward_gamma = reward_gamma

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0), requires_grad=True)
        prob = self._policy(state)
        action = prob.multinomial(1).data.numpy()[0, 0]
        return action, prob

    def loss(self, state_dict, b_s, b_a, advantage):
        prob_old = self._policy(b_s).gather(1, b_a).data
        new_model = copy.deepcopy(self._policy)
        new_model.load_state_dict(state_dict)
        prob_new = new_model(b_s).gather(1, b_a).data
        return -torch.mean((prob_new / (prob_old + 1e-8)) * advantage)

    def learn(self, env, max_iter, batch_size, sample_episodes):
        for i_iter in xrange(max_iter):
            # sample trajectories
            trajectories = [[], [], [], []]  # s, a, r, p
            for _ in xrange(sample_episodes):
                s = env.reset()
                episode_len = 0
                done = False
                while not done:
                    episode_len += 1
                    a, p = self.act(s)
                    s_, r, done, info = env.step(a)
                    trajectories[0].append(s)
                    trajectories[1].append([a])
                    trajectories[2].append([r])
                    trajectories[3].append(p)
                    s = s_
                for i in xrange(1, episode_len):
                    trajectories[2][-i-1][0] += trajectories[2][-i][0] * self.reward_gamma
            entropy = -sum((p * p.log()).sum() for p in trajectories[3]) / len(trajectories[3])

            # batch training
            for index in range(0, len(trajectories[0]), batch_size):
                b_s, b_r, b_a, b_p = (trajectories[i][index:index+batch_size] for i in xrange(4))
                b_s, b_r = map(torch.FloatTensor, [b_s, b_r])
                b_a = torch.LongTensor(b_a)
                baseline = self._value.forward(Variable(b_s)).data
                advantage = b_r - baseline
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # update value
                pass

                # update policy
                pass
