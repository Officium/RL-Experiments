import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.distributions
import torch.nn as nn

from common.logger import get_logger
from common.replay_buffer import VecReplayBuffer
from common.util import scale_ob


def learn(device,
          env, nenv, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval, ob_scale,
          gamma, grad_norm, timesteps_per_batch,
          ent_coef, vf_coef, learning_starts, buffer_size,
          replay_ratio, c, trust_region, max_kl, alpha):

    """
    Papers
    Wang Z, Bapst V, Heess N, et al. Sample efficient actor-critic with
    experience replay[J]. arXiv preprint arXiv:1611.01224, 2016.

    Parameters:
    ----------
    gram_norm (float | None): grad norm
    timesteps_per_batch (int): number of steps per update
    ent_coef (float): policy entropy coefficient in the objective
    vf_coef (float): value function loss coefficient in the objective
    learning_starts (int): how many steps of the model to collect transitions
                           for before learning starts
    buffer_size (int): size of the replay buffer
    replay_ratio (int): how many (on average) batches of data to sample from
                        the replay buffer take after batch from the environment
    c (float): importance weight clipping factor
    trust_region (bool): whether or not algorithms use KL to determine step size
    max_kl (float): max KL divergence between the old policy and updated policy
    alpha (float): momentum factor in the Polyak (exponential moving average)
                   averaging of the model parameters
    """
    name = '{}_{}'.format(os.path.split(__file__)[-1][:-3], seed)
    logger = get_logger(name)

    policy = network.to(device)
    buffer = VecReplayBuffer(env, timesteps_per_batch, buffer, device)


def _generate(device, env, policy, ob_scale,
              number_timesteps, timesteps_per_batch):
    """ Generate trajectories """
    o = env.reset()
    nc = env.observation_space.shape // env.k
    b_enc_o = np.split(o, env.k, 1)
    b_a, b_r, b_p, b_done, infos = [], [], [], [], []
    for n in range(1, number_timesteps + 1):
        # sample action
        with torch.no_grad():
            p, _, _ = policy(scale_ob(o, device, ob_scale))
            a = p.multinomial(1).cpu().numpy()
            p = p.cpu().numpy()

        # take action in env
        o_, r, done, info = env.step(a)
        for d in info:
            if d.get('episode'):
                infos.append(d['episode'])

        # store batch data and update observation
        b_enc_o.append(o_[:, -nc:])
        b_a.append(a)
        b_r.append(r)
        b_p.append(p)
        b_done.append(done)
        if n % timesteps_per_batch == 0:
            yield (
                np.asarray(b_enc_o).swapaxes(1, 0),
                np.asarray(b_a).swapaxes(1, 0),
                np.asarray(b_r).swapaxes(1, 0),
                np.asarray(b_p).swapaxes(1, 0),
                np.asarray(b_done).swapaxes(1, 0),
                np.asarray(b_done).swapaxes(1, 0)[:, 1:],
                infos
            )
            for l in (b_enc_o, b_a, b_r, b_p, b_done, infos):
                l.clear()
        o = o_
