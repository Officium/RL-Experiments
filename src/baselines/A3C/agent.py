# -*- coding: utf-8 -*-
import copy
import time

import torch
from torch.autograd import Variable
import torch.multiprocessing as mp

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Mnih V, Badia A P, Mirza M, et al. Asynchronous methods for deep reinforcement learning[C]//
    #     International Conference on Machine Learning. 2016: 1928-1937.
    def __init__(self, ac, optimizer, loss, reward_gamma=0.99, c1=0.1, c2=0):
        """
        Args:
            ac: ac network
            optimizer: optimizer for ac
            loss: loss function for value, calculate loss by `loss(eval, target)`
            reward_gamma: reward discount
            c1: coff of value, final loss is policy_loss + c1 * value_loss + c2 * entropy
            c2: coff of entropy, final loss is policy_loss + c1 * value_loss + c2 * entropy
        """
        self._ac = ac
        self.optimizer = optimizer
        self.loss = loss
        self.reward_gamma = reward_gamma
        self._c1 = c1
        self._c2 = c2

        self.max_iter = 0
        self._ac_copy = copy.deepcopy(ac)

    def act(self, state, step=None, noise=None, train=False):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        prob, value = self._ac_copy(state) if train else self._ac(state)
        action = prob.multinomial(1).data.numpy()[0, 0]
        return action, prob, value

    def evaluator(self, env, counter, episode_interval, seed):
        torch.manual_seed(seed)
        env.seed(seed)
        state = env.reset()
        e_reward = 0
        while True:
            action, _, _ = self.act(state, train=False)
            state, reward, done, _ = env.step(action)
            e_reward += reward
            if done:
                if counter.value >= self.max_iter:
                    return logger.info('Iter: {}, E_Reward {}'.format(counter.value, round(e_reward, 2)))
                else:
                    logger.info('Iter: {}, E_Reward {}'.format(counter.value, round(e_reward, 2)))
                    e_reward = 0
                    state = env.reset()
                    time.sleep(episode_interval)

    def learner(self, env, max_iter, lock, counter, seed):
        torch.manual_seed(seed)
        env.seed(seed)
        state = env.reset()
        while counter.value < self.max_iter:
            # Sync with the ac
            self._ac_copy.load_state_dict(self._ac.state_dict())

            # sample episodes
            done = False
            b_s, b_r, b_s_, b_logp, b_v, b_e = [], [], [], [], [], []
            j_iter = 0
            while (not done) and j_iter < max_iter:
                j_iter += 1
                b_s.append(state)
                action, probs, value = self.act(state, train=True)
                state, reward, done, _ = env.step(action)
                b_s_.append(state)
                b_r.append(reward)
                b_logp.append(probs.log()[0, action])
                b_v.append(value)
                b_e.append(-(probs * probs.log()).sum(1, keepdim=True))
            with lock:
                counter.value += 1
            if done:  # if done reward=0, else set to the value of next state
                state = env.reset()
                reward = Variable(torch.zeros(1, 1))
            else:
                reward = b_v[-1]

            # update parameters
            ploss = 0
            vloss = 0
            for i in reversed(xrange(len(b_s))):
                reward = b_r[i] + self.reward_gamma * reward
                ploss -= (reward - b_v[i]) * b_logp[i]
                vloss += (reward - b_v[i]).pow(2)
            # for nn.module, `.share_memory()' while not share memory for None grads
            self.optimizer.zero_grad()
            for param, param_copy in zip(self._ac.parameters(), self._ac_copy.parameters()):
                if param._grad is None:
                    param._grad = param_copy.grad
            loss = ploss + self._c1 * vloss - self._c2 * torch.sum(torch.cat(b_e))
            loss.backward()
            self.optimizer.step()

    def learn(self, env, max_iter, actor_iter, process_num, episode_interval, seed):
        """
        Args:
            env: env
            max_iter: max_iter
            actor_iter: max_iter for actors
            process_num: how many actor_learners work
            episode_interval: sleep seconds when critic get a terminal state
            seed: random seed
        """
        self.max_iter = max_iter
        torch.manual_seed(seed)
        self._ac.share_memory()
        self.optimizer.share_memory()

        # start master
        counter = mp.Value('i', 0)
        processes = []
        p = mp.Process(target=self.evaluator,
                       args=(env, counter, episode_interval, process_num + seed))
        p.start()
        processes.append(p)

        # start workers
        lock = mp.Lock()
        for rank in xrange(process_num):
            p = mp.Process(target=self.learner,
                           args=(env, actor_iter, lock, counter, rank + seed))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
