# -*- coding: utf-8 -*-
import math

import gym
import torch
import torch.nn as nn

from baselines import A3C


class AC(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 1),
        )
        self.policy = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.LogSoftmax(1)
        )
        for attr in (self.fc, self.value, self.policy):
            nn.init.xavier_normal_(attr[0].weight)
            nn.init.constant_(attr[0].bias, 0)

    def forward(self, state):
        x = self.fc(state)
        return self.policy(x), self.value(x)


# copy from https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py
class SharedAdam(torch.optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


env = gym.make('CartPole-v0')
ac = AC(env.observation_space.shape[0], env.action_space.n)
agent = A3C.Agent(ac,
                  SharedAdam(ac.parameters(), lr=1e-3),
                  nn.MSELoss())
agent.learn(env, 200000, 200, 4, 1, 1)
