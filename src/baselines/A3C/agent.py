# -*- coding: utf-8 -*-
import copy
import time

import torch
import torch.multiprocessing as mp

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Mnih V, Badia A P, Mirza M, et al. Asynchronous methods for deep reinforcement learning[C]//
    #     International Conference on Machine Learning. 2016: 1928-1937.
    def __init__(self, ac, optimizer, loss, reward_gamma=0.99, c1=1e-1, c2=1e-2):
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
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0)
            logprob, _ = self._ac_copy(state) if train else self._ac(state)
            action = logprob.exp().multinomial(1).numpy()[0, 0]
            return action

    def evaluator(self, env, counter, episode_interval, seed):
        torch.manual_seed(seed)
        env.seed(seed)
        state = env.reset()
        e_reward = 0
        while True:
            action = self.act(state, train=False)
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
            b_s, b_a, b_r = [], [], []
            while (not done) and len(b_s) < max_iter:
                action = self.act(state, train=True)
                b_s.append(state)
                b_a.append(action)
                state, reward, done, _ = env.step(action)
                b_r.append(reward)
            with lock:
                counter.value += 1
            # if done reward=0, else set to the value of next state
            if done:
                state = env.reset()
            else:
                _, value = self._ac_copy(b_s[-1])
                b_r[-1] += self.reward_gamma * value.detach()
            episode_len = len(b_s)
            for i in range(1, episode_len):
                b_r[-i-1] += self.reward_gamma * b_r[-i]

            # update parameters
            b_s = torch.Tensor(b_s).float()
            b_a = torch.Tensor(b_a).long().unsqueeze(1)
            b_r = torch.Tensor(b_r).float().unsqueeze(1)
            logp, b_v = self._ac_copy(b_s)
            entropy = -torch.sum(logp * logp.exp()).unsqueeze(0)
            b_logp = logp.gather(1, b_a)
            ploss = -torch.sum((b_r - b_v) * b_logp).unsqueeze(0)
            vloss = self.loss(b_v, b_r).unsqueeze(0)
            # for nn.module, `.share_memory()' while not share memory for None grads
            with lock:
                self.optimizer.zero_grad()
                for param, param_copy in zip(self._ac.parameters(), self._ac_copy.parameters()):
                    if param.grad is None:
                        param._grad = param_copy.grad
                loss = torch.sum(torch.cat([ploss, self._c1 * vloss, -self._c2 * entropy]))
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
        for rank in range(process_num):
            p = mp.Process(target=self.learner,
                           args=(env, actor_iter, lock, counter, rank + seed))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
