# -*- coding: utf-8 -*-
import copy
import functools

import numpy
import torch
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from baselines import base
from utils.logger import get_logger
logger = get_logger()


class Agent(base.Agent):
    # References:
    # [1] Schulman J. Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs[D].
    #     UC Berkeley, 2016.
    # [2] https://github.com/joschu/modular_rl
    def __init__(self, policy, value, loss, optimizer, accept_ratio=0.9, reward_gamma=0.9,
                 cg_iters=10, cg_tol=1e-10, cg_damping=1e-3, max_kl=1e-2, ls_iters=10):
        """
        Args:
            policy: policy network
            value: value network (state -> value)
            loss: loss function for value, calculate loss by `loss(eval, target)`
            optimizer: optimizer for value
            accept_ratio: improvement accept ratio in linear search
            reward_gamma: reward discount
            cg_iters: max iters of cg
            cg_tol: tolerence of cg
            cg_damping: add multiple of the identity to Fisher matrix during CG
            max_kl: max KL divergence between old and new policy
            ls_iters: max backtracks of linear search
        """
        self._policy = policy
        self._value = value
        self.loss = loss
        self.optimizer = optimizer
        self.accept_ratio = accept_ratio
        self.reward_gamma = reward_gamma
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol
        self.cg_damping = cg_damping
        self.max_kl = max_kl
        self.ls_iters = ls_iters

    def act(self, state, step=None, noise=None):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0), requires_grad=True)
        prob = self._policy(state)
        action = prob.multinomial(1).data.numpy()[0, 0]
        return action, prob

    def get_loss(self, theta, b_s, b_a, advantage):
        # get surrogate loss
        prob_old = self._policy(b_s).gather(1, b_a).data
        new_model = copy.deepcopy(self._policy)
        vector_to_parameters(theta, new_model.parameters())
        prob_new = new_model(b_s).gather(1, b_a).data
        return -torch.mean((prob_new / prob_old) * advantage)

    def learn(self, env, max_iter, batch_size, sample_episodes):
        for i_iter in xrange(max_iter):
            # sample trajectories using single path
            trajectories = [[], [], [], []]  # s, a, r, p
            for _ in xrange(sample_episodes):
                s = env.reset()
                episode_len = 0
                done = False
                while not done:
                    episode_len += 1
                    a, p = self.act(s)
                    s_, r, done, info = env.step(a)
                    trajectories[0].append(s)
                    trajectories[1].append([a])
                    trajectories[2].append([r])
                    trajectories[3].append(p)
                    s = s_
                for i in xrange(1, episode_len):
                    trajectories[2][-i-1][0] += trajectories[2][-i][0] * self.reward_gamma
            entropy = -sum((p * p.log()).sum() for p in trajectories[3]) / len(trajectories[3])

            # batch training
            for index in xrange(0, len(trajectories[0]), batch_size):
                # load batch data
                b_s, b_a, b_r, b_p = (trajectories[i][index:index+batch_size] for i in xrange(4))
                b_s, b_r = map(torch.FloatTensor, [b_s, b_r])
                b_a = torch.LongTensor(b_a)
                baseline = self._value.forward(Variable(b_s))
                advantage = b_r - baseline.data
                advantage = (advantage - advantage.mean()) / advantage.std()

                # update value i.e. improve baseline
                loss = self.loss(baseline, Variable(b_r))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update policy using Fisher information matrix (Refer to section 3.12 of John's Ph.D thesis)
                pg = None  # TODO
                if numpy.allclose(pg, 0):
                    logger.warn("got zero gradient. not updating")
                else:
                    stepdir = self.cg(self.fisher_vector_product, -pg)
                    shs = 0.5 * stepdir.dot(self.fisher_vector_product(stepdir))
                    lm = numpy.sqrt(shs / self.max_kl)
                    fullstep = stepdir / lm
                    expected_improve_rate = -g.dot(stepdir) / lm
                    loss_fun = functools.partial(self.get_loss, b_s=b_s, b_a=b_a, advantage=advantage)
                    old_theta = parameters_to_vector(self._policy.parameters())
                    success, new_theta = self.line_search(loss_fun, old_theta, fullstep, expected_improve_rate)
                    vector_to_parameters(new_theta, self._policy.parameters())

    @staticmethod
    def fisher_vector_product(vec):  # TODO
        pass

    def cg(self, fvp, b):
        p = b.copy()
        r = b.copy()
        x = numpy.zeros_like(b)
        rdotr = r.dot(r)
        for i in xrange(self.cg_iters):
            z = fvp(p)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < self.cg_tol:
                break
        return x

    def line_search(self, f, x, fullstep, expected_improve_rate):
        # shrink exponentially
        fval = f(x)
        for i in xrange(self.ls_iters):
            stepfrac = 0.5 ** i
            xi = x + stepfrac * fullstep
            actual_improve = fval - f(xi)
            if actual_improve > 0:
                if actual_improve / (stepfrac * expected_improve_rate) > self.accept_ratio:
                    return True, xi
        return False, x
