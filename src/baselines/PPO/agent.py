# -*- coding: utf-8 -*-
import numpy
import torch
from torch.autograd import Variable

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J].
    #     arXiv preprint arXiv:1707.06347, 2017.
    def __init__(self, ac, loss, optimizer, epsilon=0.2, reward_gamma=0.99, c1=1e-4, c2=1e-6):
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

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        prob, value = self._ac(state)
        action = prob.multinomial(1).data.numpy()[0, 0]
        return action, prob, value

    def learn(self, env, max_iter, sample_episodes=32, optim_max_iter=4, optim_batch_size=256):
        for i_iter in xrange(max_iter):
            # sample trajectories using single path
            trajectories = [[], [], [], [], []]  # s, a, r, p, v
            e_reward = 0
            for _ in xrange(sample_episodes):
                # env.render()
                s = env.reset()
                episode_len = 0
                done = False
                while not done:
                    episode_len += 1
                    a, p, v = self.act(s)
                    s_, r, done, info = env.step(a)
                    e_reward += r
                    trajectories[0].append(s)
                    trajectories[1].append([a])
                    trajectories[2].append([r])
                    trajectories[3].append(p)
                    trajectories[4].append(v * (1 - done))
                    s = s_
                for i in xrange(1, episode_len):
                    trajectories[2][-i-1][0] += trajectories[2][-i][0] * self.reward_gamma
            e_reward /= sample_episodes

            # batch training
            n = len(trajectories[0])
            batch_size = min(optim_batch_size, n)
            for j_iter in xrange(optim_max_iter):
                # load batch data
                for indexes in numpy.array_split(numpy.random.permutation(n), n / batch_size):
                    b_s, b_a, b_r, b_p, b_v = ([trajectories[i][j] for j in indexes] for i in xrange(len(trajectories)))
                    b_s, b_r = map(torch.FloatTensor, [b_s, b_r])
                    b_a = torch.LongTensor(b_a)
                    b_p = torch.cat(b_p).gather(1, Variable(b_a))
                    b_e = -(b_p.log() * b_p).sum(-1)  # entropy
                    b_v = torch.cat(b_v)
                    advantage = b_r - b_v.data
                    advantage = (advantage - advantage.mean()) / advantage.std()

                    # update ac
                    vloss = self.loss(b_v, Variable(b_r))  # value loss, L^VF
                    advantage = Variable(advantage)
                    ratio = b_p / (b_p.detach() + 1e-8)
                    # policy loss, maybe very small because of the normalization, one can use a small self._c1 to solve
                    ploss = -torch.mean(torch.min(ratio * advantage,  # -L^CLIP
                                                  ratio.clamp(1 - self._epsilon, 1 + self._epsilon) * advantage))
                    loss = ploss + self._c1 * vloss - self._c2 * b_e.mean()  # the same as openai/baselines's code
                    self.optimizer.zero_grad()
                    # retain the graph because different j_iter may use same trajectory
                    loss.backward(retain_graph=j_iter < optim_max_iter-1)
                    self.optimizer.step()
            logger.info('Iter: {}, E_Reward: {}'.format(i_iter, round(e_reward, 2)))
