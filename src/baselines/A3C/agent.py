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
    def __init__(self, policy, value, optimizer_policy, optimizer_value, loss, reward_gamma):
        """
        Args:
            policy: policy network
            value: value network (state -> value)
            optimizer_policy: optimizer for policy
            optimizer_value: optimizer for value
            loss: loss function for value, calculate loss by `loss(eval, target)`
            reward_gamma: reward discount
        """
        self._policy = policy
        self._value = value
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.loss = loss
        self.reward_gamma = reward_gamma

        self.max_iter = 0
        self._policy_copy = copy.deepcopy(policy)
        self._value_copy = copy.deepcopy(value)

    def act(self, state, step=None, noise=None, train=False):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0), requires_grad=True)
        prob = self._policy_copy(state) if train else self._policy(state)
        action = prob.multinomial(1).data.numpy()[0, 0]
        return action, prob

    def critic_learner(self, env, counter, episode_interval, seed):
        torch.manual_seed(seed)
        env.seed(seed)
        state = env.reset()
        e_reward = 0
        while counter.value < self.max_iter:
            action, _ = self.act(state, train=False)
            state, reward, done, _ = env.step(action)
            e_reward += reward
            if done:
                logger.info('Iter: {}, E_Reward {}'.format(counter.value, round(e_reward, 2)))
                e_reward = 0
                state = env.reset()
                time.sleep(episode_interval)

    def actor_learner(self, env, max_iter, lock, counter, seed):
        torch.manual_seed(seed)
        env.seed(seed)
        state = env.reset()
        while counter.value < self.max_iter:
            # Sync with the ac
            self._policy_copy.load_state_dict(self._policy.state_dict())
            self._value_copy.load_state_dict(self._value.state_dict())

            done = False
            j_iter = 0
            states, rewards, log_probs = [], [], []
            while (not done) and (j_iter < max_iter):
                action, probs = self.act(state, train=True)
                state, reward, done, _ = env.step(action)
                states.append(state)
                rewards.append(reward)
                log_probs.append(probs.log()[0, action])
                j_iter += 1
                with lock:
                    counter.value += 1

            if done:
                state = env.reset()
                reward = Variable(torch.zeros(1, 1))
            else:
                reward = self._value_copy(Variable(torch.unsqueeze(torch.FloatTensor(state), 0)))

            policy_loss = 0
            value_loss = 0
            for i in reversed(xrange(j_iter)):
                reward = rewards[i] + self.reward_gamma * reward
                value = self._value_copy(Variable(torch.unsqueeze(torch.FloatTensor(states[i]), 0)))
                policy_loss += (reward - value) * log_probs[i]
                value_loss += 0.5 * (reward - value).pow(2)

            # update policy
            self.optimizer_policy.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.optimizer_policy.step()
            # update value
            self.optimizer_value.zero_grad()
            value_loss.backward(retain_graph=True)
            self.optimizer_value.step()

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
        self._policy.share_memory()
        self._value.share_memory()
        self.optimizer_policy.share_memory()
        self.optimizer_value.share_memory()

        # start master
        counter = mp.Value('i', 0)
        processes = []
        p = mp.Process(target=self.critic_learner,
                       args=(env, counter, episode_interval, process_num + seed))
        p.start()
        processes.append(p)

        # start workers
        lock = mp.Lock()
        for rank in xrange(process_num):
            p = mp.Process(target=self.actor_learner,
                           args=(env, actor_iter, lock, counter, rank + seed))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
