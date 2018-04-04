# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

from baselines import base
from utils.logger import get_logger

logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Williams R J. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning[J].
    #     Machine Learning, 1992: 229-256.
    def __init__(self, policy, optimizer, reward_gamma=0.9):
        """ This algorithm is also known as likelihood ratio policy gradient or vanilla policy gradients
        Args:
            policy: policy network
            optimizer: optimizer for value
            reward_gamma: reward discount
        """
        self._policy = policy
        self.optimizer = optimizer
        self.reward_gamma = reward_gamma

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        prob = self._policy(state)
        action = prob.multinomial(1).data.numpy()[0, 0]
        return action, prob[0][action]

    def learn(self, env, max_iter, batch_size):
        """
        Args:
            env: env
            max_iter: max_iter
            batch_size: how many episodes to sample in each iteration
        """
        for i_iter in xrange(max_iter):
            e_reward = 0
            for j_iter in xrange(batch_size):
                # env.render()
                s = env.reset()
                log_probs = []
                rewards = []
                done = False
                while not done:
                    a, prob = self.act(s)
                    s_, r, done, info = env.step(a)
                    log_probs.append(prob.log())
                    rewards.append(r)
                    e_reward += r
                    s = s_
                episode_len = len(rewards)
                loss = 0
                for t in xrange(1, episode_len):
                    rewards[episode_len - t - 1] += rewards[episode_len - t] * self.reward_gamma
                    loss -= log_probs[episode_len - t - 1] * rewards[episode_len - t - 1]  # likelihood ratio
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            e_reward /= batch_size
            logger.info('Iter: {}, E_Reward: {}'.format(i_iter, round(e_reward, 2)))
