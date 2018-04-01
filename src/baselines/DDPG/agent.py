# -*- coding: utf-8 -*-
import numpy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J].
    #     arXiv preprint arXiv:1509.02971, 2015.
    def __init__(self,
                 actor, target_actor, optimizer_actor,
                 critic, target_critic, optimizer_critic,
                 replay_module, noise_generator, reward_gamma=0.99, tau=1e-3, warmup_size=100):
        """ Just follow the `Algorithm 1` in [1], suppose any element of action in [-1, 1]
        Args:
            actor: actor network
            target_actor: target actor network
            optimizer_actor: optimizer of actor, e.g. torch.optim.Adam
            critic: critic network
            target_critic: target critic network
            optimizer_critic: optimizer of critic, e.g. torch.optim.Adam
            replay_module: replay buffer
            noise_generator: random process for action exploration
            reward_gamma: reward discount
            tau: soft update parameter of target network, i.e. theta^target = /tau * theta + (1 - /tau) * theta^target
            warmup_size: no training until the length of replay module is larger than `warmup_size`
        """
        self._actor = actor
        self._target_actor = target_actor
        self._optimizer_actor = optimizer_actor
        self._critic = critic
        self._target_critic = target_critic
        self._optimizer_critic = optimizer_critic
        self._replay_module = replay_module
        self._noise_generator = noise_generator
        self.reward_gamma = reward_gamma
        self.tau = tau
        self.warmup_size = warmup_size

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        action = self._actor.forward(state).data
        if noise is not None:
            action += noise
        return action.clamp(-1, 1).numpy()[0]

    def learn(self, env, max_iter, batch_size):
        for i_iter in xrange(max_iter):
            s = env.reset()
            self._noise_generator.reset()
            done = False
            e_reward = 0
            while not done:
                # env.render()
                noise = torch.FloatTensor(self._noise_generator.generate())
                a = self.act(s, noise=noise)
                s_, r, done, info = env.step(a)
                e_reward += r
                self._replay_module.add(tuple((s, a, [r], s_)))

                if len(self._replay_module) < self.warmup_size:
                    continue
                # sample batch transitions
                b_s, b_a, b_r, b_s_ = self._replay_module.sample(batch_size)
                b_s = numpy.vstack(b_s)
                b_a = numpy.vstack(b_a)
                b_s, b_a, b_r = map(lambda ryo: Variable(torch.FloatTensor(ryo)), [b_s, b_a, b_r])
                b_s_ = Variable(torch.FloatTensor(b_s_), volatile=True)

                # update critic
                self._optimizer_critic.zero_grad()
                y = b_r + self.reward_gamma * self._target_critic.forward(b_s_, self._target_actor.forward(b_s_))
                loss = nn.MSELoss()(self._critic.forward(b_s, b_a), y)
                loss.backward()
                self._optimizer_critic.step()

                # update actor
                self._optimizer_actor.zero_grad()
                loss = -self._critic.forward(b_s, self._actor.forward(b_s)).mean()
                loss.backward()
                self._optimizer_actor.step()

                # update target networks
                for target, normal in [(self._target_actor, self._actor), (self._target_critic, self._critic)]:
                    target_vec = parameters_to_vector(target.parameters())
                    normal_vec = parameters_to_vector(normal.parameters())
                    vector_to_parameters((1 - self.tau) * target_vec + self.tau * normal_vec, target.parameters())
            logger.info('Iter: {}, E_Reward: {}'.format(i_iter, round(e_reward, 2)))
