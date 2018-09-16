# -*- coding: utf-8 -*-
import numpy as np
import torch

from baselines import base
from utils.logger import get_logger
logger = get_logger()
logger.warn('This is a one-step version without prioritized replay buffer!'
            'Games like CartPole can not be learned well because of one-step distributional RL.')


class Agent(base.Agent):
    # References:
    # [1] Hessel M, Modayil J, Van Hasselt H, et al.
    #     Rainbow: Combining Improvements in Deep Reinforcement Learning[J]. 2017.
    def __init__(self, net, target_net, replay_module, optimizer, target_replace_iter,
                 value_min, value_max, atom_num, min_replay_buffer, reward_gamma=1.0):
        """ Rainbow
        Args:
            net: Rainbow network
            target_net: Target Rainbow network
            replay_module: replay module
            optimizer: optimizer, e.g. torch.optim.Adam
            target_replace_iter: replace target q network by q network every which iters
            value_min: min value
            value_max: max value
            atom_num: atom number
            min_replay_buffer: when replay buffer's length is less than this value, don't update networks
            reward_gamma: reward discount
        """
        self._net = net
        self._target_net = target_net
        self._replay_module = replay_module
        self.optimizer = optimizer
        self._value_min = value_min
        self._value_max = value_max
        self._atom_num = atom_num
        self._min_replay_buffer = min_replay_buffer
        self.target_replace_iter = target_replace_iter
        self.reward_gamma = reward_gamma

    def act(self, state, step=None, noise=None):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            distribution = self._net(state).exp().cpu().numpy()
            distribution *= np.linspace(self._value_min, self._value_max, self._atom_num)
            return distribution.sum(2).argmax(1)[0]

    def learn(self, env, max_iter, batch_size):
        learn_counter = 0
        for i_iter in range(max_iter):
            s = env.reset()
            e_reward = 0
            done = False
            while not done:
                # env.render()
                a = self.act(s)
                s_, r, done, info = env.step(a)
                self._replay_module.add(tuple((s, [a], [r], s_, [int(done)])))
                s = s_
                e_reward += r

                # update target q network
                if learn_counter % self.target_replace_iter == 0:
                    self._target_net.load_state_dict(self._net.state_dict())
                    for param in self._target_net.parameters():
                        param.requires_grad = False
                learn_counter += 1

                if len(self._replay_module) <= self._min_replay_buffer:
                    continue

                # sample batch transitions
                b_s, b_a, b_r, b_s_, b_d = map(torch.Tensor, self._replay_module.sample(batch_size))
                b_a = b_a.long()

                # categorical algorithm
                b_pia = self._target_net(b_s_).exp()  # batch p_i(s_, a), b * action_dim * atom_num
                delta_z = float(self._value_max - self._value_min) / (self._atom_num - 1)
                z_i = torch.linspace(self._value_min, self._value_max, self._atom_num)
                b_q_ = torch.bmm(b_pia, z_i.unsqueeze(1).repeat(batch_size, 1, 1))  # Q(s_, a), b * action_dim * 1
                b_a_ = b_q_.max(1)[1]
                b_tzj = (self.reward_gamma * (1 - b_d) * z_i.unsqueeze(0).repeat(batch_size, 1) +
                         b_r.repeat(1, self._atom_num)).clamp(self._value_min, self._value_max)  # b * atom_num
                b_b = (b_tzj - self._value_min) / delta_z
                b_l = b_b.floor()
                b_u = b_b.ceil()
                b_pia_ = b_pia[torch.arange(batch_size), b_a_.squeeze(), :]  # b * atom_num
                b_m = torch.zeros(batch_size, self._atom_num)
                b_m.scatter_add_(1, b_l.long(), b_pia_ * (b_u - b_b))
                b_m.scatter_add_(1, b_u.long(), b_pia_ * (b_b - b_l))
                loss = -self._net(b_s)[torch.arange(batch_size), b_a.squeeze(), :].squeeze(1).mul(b_m).sum(1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # reset noise
                self._net.reset_noise()
                self._target_net.reset_noise()
            logger.info('Iter: {}, E_Reward: {}'.format(i_iter, round(e_reward, 2)))
