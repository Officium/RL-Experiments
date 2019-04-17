import os

import torch
from torch.utils.data import DataLoader

from common.logger import get_logger
from common.models import build_policy_with_value, get_optimizer


def learn(device,
          env, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval,
          timesteps_per_batch, cg_iters, cg_damping,
          max_kl, lam, vf_iters, vf_stepsize, entcoeff, **kwargs):
    """
    Thesis:
    Schulman J. Optimizing Expectations: From Deep Reinforcement Learning to
    Stochastic Computation Graphs[D]. UC Berkeley, 2016.

    Official implementation:
    https://github.com/joschu/modular_rl

    Parameters:
    ----------
        timesteps_per_batch (int): timesteps per gradient estimation batch
        cg_iters (int): number of iterations of conjugate gradient algorithm
        cg_damping (float): conjugate gradient damping
        max_kl (float): max KL(pi_old || pi)
        lam (float): advantage estimation
        vf_iters (float): number of value update per policy update
        vf_stepsize (int): learning rate for optimizer of value
        entcoeff (float): coefficient of policy entropy term
    """
    pass
