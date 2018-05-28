# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J].
    #     arXiv preprint arXiv:1707.06347, 2017.
    def __init__(self, ac, loss, optimizer, epsilon=0.2,
                 reward_gamma=0.99, c1=1e-4, c2=1e-6, gae_lambda=0.95):
        """
        Args:
            ac: ac network (state -> prob, value)
            loss: loss function for value, calculate loss by `loss(eval, target)`
            optimizer: optimizer for ac
            epsilon: epsilon in clip
            reward_gamma: reward discount
            c1: factor of value loss
            c2: factor of entropy
        """
        self._ac = ac
        self.loss = loss
        self.optimizer = optimizer
        self._epsilon = epsilon
        self.reward_gamma = reward_gamma
        self._c1 = c1
        self._c2 = c2
        self.gae_lambda = gae_lambda

    def act(self, state, step=None, noise=None):
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0)
            logprob, value = self._ac(state)
            action = logprob.exp().multinomial(1).numpy()[0, 0]
            return action, logprob[0, action], value[0, 0]

    def learn(self, env, max_iter, sample_episodes=32, optim_max_iter=4, optim_batch_size=256):
        for i_iter in range(max_iter):
            # sample trajectories using single path
            trajectories = []  # s, a, r, logp
            e_reward = 0
            for _ in range(sample_episodes):
                # env.render()
                values = []
                s = env.reset()
                done = False
                while not done:
                    a, logp, v = self.act(s)
                    s_, r, done, _ = env.step(a)
                    e_reward += r
                    trajectories.append([s, a, r, logp])
                    values.append(v)
                    s = s_
                episode_len = len(values)
                gae = np.empty(episode_len)
                gae[-1] = last_gae = trajectories[-1][2] - values[-1]
                for i in range(1, episode_len):
                    delta = trajectories[-i-1][2] + self.reward_gamma * values[-i] - values[-i-1]
                    gae[-i-1] = last_gae = delta + self.reward_gamma * self.gae_lambda * last_gae
                for i in range(episode_len):
                    trajectories[-(episode_len-i)][2] = gae[i] + values[i]
            e_reward /= sample_episodes

            # batch training
            batch_size = min(optim_batch_size, len(trajectories))
            for j_iter in range(optim_max_iter):
                # load batch data
                loader = DataLoader(trajectories, batch_size=batch_size, shuffle=True)
                for b_s, b_a, b_r, b_logp_old in loader:
                    b_s = b_s.float()
                    b_a = b_a.long().unsqueeze(1)
                    b_r = b_r.float().unsqueeze(1)
                    b_logp_old = b_logp_old.float().unsqueeze(1)
                    b_logp, b_v = self._ac(b_s)
                    entropy = -(b_logp * b_logp.exp()).sum(-1).mean()
                    b_logp = b_logp.gather(1, b_a)
                    advantage = b_r - b_v
                    advantage = (advantage - advantage.mean()) / advantage.std()

                    # update ac
                    vloss = self.loss(b_v, b_r)  # value loss, L^VF
                    ratio = (b_logp - b_logp_old).exp()
                    # policy loss, maybe very small because of the normalization, one can use a small self._c1 to solve
                    clip = torch.min(ratio * advantage,  # L^CLIP
                                     ratio.clamp(1 - self._epsilon, 1 + self._epsilon) * advantage).mean()
                    loss = -clip + self._c1 * vloss - self._c2 * entropy  # the same as openai/baselines's code
                    self.optimizer.zero_grad()
                    # retain the graph because different j_iter may use same trajectory
                    loss.backward()
                    self.optimizer.step()
            logger.info('Iter: {}, E_Reward: {}'.format(i_iter, round(e_reward, 2)))
