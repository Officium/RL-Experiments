# -*- coding: utf-8 -*-
import random

import numpy
import torch
from torch.autograd import Variable

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    def __init__(self, q, target_q, replay_module, optimizer, loss,
                 target_replace_iter, greedy_epsilon=0.9, reward_gamma=0.9):
        """
        Args:
            q: Q-network
            target_q: Target-Q-network
            replay_module: replay module
            optimizer: optimizer, e.g. torch.optim.Adam
            loss: loss function, e.g. torch.nn.MSELoss
            target_replace_iter: replace target q network by q network every which iters
            greedy_epsilon: greedy rate
            reward_gamma: reward discount
        """
        self._q = q
        self._target_q = target_q
        self._replay_module = replay_module
        self.optimizer = optimizer
        self.loss = loss
        self.epsilon = greedy_epsilon
        self.target_replace_iter = target_replace_iter
        self.reward_gamma = reward_gamma

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        value = self._q.forward(state)
        if random.random() < self.epsilon:  # greedy
            return numpy.argmax(value[0].data.numpy())
        else:  # random
            return random.randrange(value.size(1))

    def learn(self, env, max_iter, batch_size):
        learn_counter = 0
        for i_episode in xrange(max_iter):
            s = env.reset()
            e_reward = 0
            done = False
            while not done:
                # env.render()
                a = self.act(s)
                s_, r, done, info = env.step(a)
                self._replay_module.add(tuple((s, [a], [r], s_)))
                e_reward += r

                # update target q network
                if learn_counter % self.target_replace_iter == 0:
                    self._target_q.load_state_dict(self._q.state_dict())
                learn_counter += 1

                # sample batch transitions
                b_s, b_a, b_r, b_s_ = self._replay_module.sample(batch_size)
                b_s, b_r, b_s_ = map(torch.FloatTensor, [b_s, b_r, b_s_])
                b_a = torch.LongTensor(b_a)
                b_s, b_a, b_r, b_s_ = map(Variable, [b_s, b_a, b_r, b_s_])

                # update parameters
                q_eval = self._q(b_s).gather(1, b_a)  # shape (batch, 1)
                q_next = self._target_q(b_s_).detach()  # detach from graph, don't backpropagate
                q_target = b_r + self.reward_gamma * q_next.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
                loss = self.loss(q_eval, q_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if done:
                    logger.info('Iter: {}, E_Reward: {}'.format(i_episode, round(e_reward, 2)))
                s = s_
