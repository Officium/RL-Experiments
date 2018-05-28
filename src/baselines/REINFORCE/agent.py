# -*- coding: utf-8 -*-
import torch

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
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0)
            logprob = self._policy(state)
            action = logprob.exp().multinomial(1).numpy()[0, 0]
            return action

    def learn(self, env, max_iter, batch_size):
        """
        Args:
            env: env
            max_iter: max_iter
            batch_size: how many episodes to sample in each iteration
        """
        for i_iter in range(max_iter):
            e_reward = 0
            for j_iter in range(batch_size):
                # env.render()
                s = env.reset()
                b_s, b_a, b_r = [[], [], []]  # s, a, r
                done = False
                while not done:
                    a = self.act(s)
                    s_, r, done, info = env.step(a)
                    b_s.append(s)
                    b_a.append(a)
                    b_r.append(r)
                    e_reward += r
                    s = s_
                episode_len = len(b_s)
                for t in range(1, episode_len):
                    b_r[episode_len-t-1] += b_r[episode_len - t] * self.reward_gamma
                b_s = torch.Tensor(b_s)
                b_a = torch.Tensor(b_a).long().unsqueeze(1)
                b_r = torch.Tensor(b_r).unsqueeze(1)
                b_logp = self._policy(b_s).gather(1, b_a)
                loss = -(b_logp * b_r).sum()  # likelihood ratio
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            e_reward /= batch_size
            logger.info('Iter: {}, E_Reward: {}'.format(i_iter, round(e_reward, 2)))
