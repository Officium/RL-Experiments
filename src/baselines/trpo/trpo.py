import copy
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.nn.utils.convert_parameters import parameters_to_vector

from common.logger import get_logger
from common.models import build_policy, build_value, get_optimizer
from common.util import Trajectories


def learn(device,
          env, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval,
          gamma, timesteps_per_batch, cg_iters, cg_damping,
          max_kl, gae_lam, vf_iters, vf_lr, entcoeff, **kwargs):
    """
    Thesis:
    Schulman J. Optimizing Expectations: From Deep Reinforcement Learning to
    Stochastic Computation Graphs[D]. UC Berkeley, 2016.

    Official implementation:
    https://github.com/joschu/modular_rl

    There are a little differences between openai's implementation and official
    one in policy update. We follow the openai's version which is also committed
    by John Schulman.

    Parameters:
    ----------
        timesteps_per_batch (int): timesteps per gradient estimation batch
        cg_iters (int): number of iterations of conjugate gradient algorithm
        cg_damping (float): conjugate gradient damping
        max_kl (float): max KL(pi_old || pi)
        gae_lam (float): advantage estimation
        vf_iters (float): number of value update per policy update
        vf_lr (int): learning rate for optimizer of value
        entcoeff (float): coefficient of policy entropy term

    """
    name = '{}_{}'.format(os.path.split(__file__)[-1][:-3], seed)
    logger = get_logger(name)
    logger.critical('TRPO is not implemented! Welcome any contributions~')
