# -*- coding: utf-8 -*-
import random

import torch
from torch.autograd import Variable

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J].
    #     arXiv preprint arXiv:1707.06347, 2017.
    def __init__(self, policy, value, loss, optimizer_policy, optimizer_value,
                 epsilon=0.2, reward_gamma=0.9):
        """
        Args:
            policy: policy network
            value: value network (state -> value)
            loss: loss function for value, calculate loss by `loss(eval, target)`
            optimizer_policy: optimizer for policy
            optimizer_value: optimizer for value
            epsilon: epsilon in clip
            reward_gamma: reward discount
        """
        self._policy = policy
        self._value = value
        self.loss = loss
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self._epsilon = epsilon
        self.reward_gamma = reward_gamma

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0), requires_grad=True)
        prob = self._policy(state)
        action = prob.multinomial(1).data.numpy()[0, 0]
        return action, prob

    def learn(self, env, max_iter, sample_episodes=256, optim_max_iter=4, optim_batch_size=64):
        for i_iter in xrange(max_iter):
            # sample trajectories using single path
            trajectories = [[], [], [], []]  # s, a, r, p
            e_reward = 0
            for _ in xrange(sample_episodes):
                # env.render()
                s = env.reset()
                episode_len = 0
                done = False
                reward = 0.0
                while not done:
                    episode_len += 1
                    a, p = self.act(s)
                    s_, r, done, info = env.step(a)
                    reward += r
                    trajectories[0].append(s)
                    trajectories[1].append([a])
                    trajectories[2].append([r])
                    trajectories[3].append(p)
                    s = s_
                for i in xrange(1, episode_len):
                    trajectories[2][-i-1][0] += trajectories[2][-i][0] * self.reward_gamma
                e_reward += reward / episode_len
            e_reward /= sample_episodes

            # batch training
            for j_iter in xrange(optim_max_iter):
                # load batch data
                indexes = random.sample(xrange(len(trajectories[0])), optim_batch_size)
                b_s, b_a, b_r, b_p = ([trajectories[i][j] for j in indexes] for i in xrange(4))
                b_s, b_r = map(torch.FloatTensor, [b_s, b_r])
                b_p = torch.cat(b_p)
                baseline = self._value.forward(Variable(b_s))
                advantage = b_r - baseline.data
                advantage = (advantage - advantage.mean()) / advantage.std()

                # update value
                loss = self.loss(baseline, Variable(b_r))
                self.optimizer_value.zero_grad()
                loss.backward()
                self.optimizer_value.step()

                # update policy
                advantage = Variable(advantage)
                ratio = b_p / b_p.detach()
                loss = -torch.mean(torch.min(ratio * advantage,
                                             ratio.clamp(1 - self._epsilon, 1 + self._epsilon) * advantage))
                self.optimizer_policy.zero_grad()
                loss.backward(retain_graph=j_iter < optim_max_iter - 1)
                self.optimizer_policy.step()
            logger.info('Iter: {}, E_Reward: {}'.format(i_iter, round(e_reward, 2)))
