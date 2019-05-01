import os
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from common.logger import get_logger
from common.replay_buffer import ReplayBuffer
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

    Note that we use lazy frame instead of interval variable `enc_obs` in
    openai's implementation to save memory.

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
    polyak_policy = deepcopy(policy)
    buffer = ReplayBuffer(buffer_size, device)
    number_timesteps = number_timesteps // nenv
    generator = _generate(
        device, env, nenv, policy, ob_scale,
        number_timesteps, timesteps_per_batch
    )
    max_iter = number_timesteps // timesteps_per_batch
    scheduler = LambdaLR(optimizer, lambda i_iter: 1 - i_iter / max_iter)

    eps = 1e-6
    total_timesteps = 0
    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}
    start_ts = time.time()
    for n_iter in range(1, max_iter + 1):
        scheduler.step()

        batch = generator.__next__()
        *data, info = batch
        assert len(data) == nenv
        total_timesteps += timesteps_per_batch
        for d in info:
            infos['eplenmean'].append(d['l'])
            infos['eprewmean'].append(d['r'])
        for env_data in data:
            buffer.add(*env_data)
        b_o = [x[0] for x in data]
        b_a = [x[1] for x in data]
        b_r = [x[2] for x in data]
        b_o_ = [x[3] for x in data]
        b_d = [x[4] for x in data]
        b_logp = [x[5] for x in data]
        mode_sample = [  # size (nenv, nstep, *)
            ('on-policy', (
                torch.from_numpy(np.asarray(b_o)).to(device).float(),
                torch.from_numpy(np.asarray(b_a)).to(device).long(),
                torch.from_numpy(np.asarray(b_r)).to(device).float(),
                torch.from_numpy(np.asarray(b_o_)).to(device).float(),
                torch.from_numpy(np.asarray(b_d)).to(device).float(),
                torch.from_numpy(np.asarray(b_logp)).to(device).float()))
        ] + [('off-policy', buffer.sample(nenv))
             for _ in range(np.random.poisson(replay_ratio))]

        flat = lambda tensor: tensor.view(nenv * timesteps_per_batch, -1)
        unflat = lambda tensor: tensor.view(nenv, timesteps_per_batch, -1)
        for mode, (b_o, b_a, b_r, b_o_, b_done, b_logp_old) in mode_sample:
            b_logp, b_q, b_v = policy(flat(b_o) * ob_scale)
            _, _, b_v_ = policy(flat(b_o_) * ob_scale)
            b_logp_poly, b_q_poly, b_v_poly = polyak_policy(flat(b_o))
            b_p = b_logp.exp()
            b_p_old = flat(b_logp_old.exp())
            b_p_i = b_p.gather(1, flat(b_a))
            b_logp_i = b_logp.gather(1, flat(b_a))
            b_q_i = b_q.gather(1, flat(b_a))

            entropy = -(b_p * b_logp).sum(-1).mean()

            rho = b_p / (b_p_old + eps)
            rho_i = b_p_i / (b_p_old.gather(1, flat(b_a)) + eps)
            q_retrace = _q_retrace(b_r, b_done, unflat(b_q_i),
                                   unflat(b_v_), unflat(rho_i),
                                   timesteps_per_batch, gamma)
            adv = unflat(q_retrace) - b_v
            loss_is = -(b_logp_i * (adv * rho_i.clamp(None, c)).detach()).mean()

            adv_bc = (b_q - b_v)
            rho_t = (1 - (c / (eps + rho))).clamp(0)
            loss_bc = -(b_logp * (adv_bc * rho_t).detach()).sum(1).mean()

            loss_q = (0.5 * (b_q.gather(1, b_a) - q_retrace.detach())).mean()

            if trust_region:
                loss = loss_is + loss_bc - ent_coef * entropy
                loss *= timesteps_per_batch * nenv
                var = policy.parameters()
                grad = torch.autograd.grad(-loss, var)
                raise ValueError
            else:
                loss = loss_is + loss_bc + vf_coef * loss_q - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                if grad_norm is not None:
                    nn.utils.clip_grad_norm_(policy.parameters(), grad_norm)
                optimizer.step()
            vec_polyak = parameters_to_vector(polyak_policy.parameters())
            vec_policy = parameters_to_vector(policy.parameters())
            vec_polyak = alpha * vec_polyak + (1 - alpha) * vec_policy
            vector_to_parameters(vec_polyak, polyak_policy.parameters())

        logger.info('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
        fps = int(total_timesteps / (time.time() - start_ts))
        logger.info('Total timesteps {} FPS {}'.format(total_timesteps, fps))
        for k, v in infos.items():
            v = (sum(v) / len(v)) if v else float('nan')
            logger.info('{}: {:.6f}'.format(k, v))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.{}'.format(name, n_iter)))


def _generate(device, env, nenv, policy, ob_scale,
              number_timesteps, timesteps_per_batch):
    """ Generate trajectories """
    o = env.reset()
    b_o, b_a, b_r, b_o_, b_d, b_logp, infos = [], [], [], [], [], [], []
    for n in range(1, number_timesteps + 1):
        # sample action
        with torch.no_grad():
            logp, _, _ = policy(scale_ob(o, device, ob_scale))
            a = logp.exp().multinomial(1).cpu().numpy()[:, 0]
            logp = logp.cpu().numpy()

        # take action in env
        o_, r, done, info = env.step(a)
        for d in info:
            if d.get('episode'):
                infos.append(d['episode'])

        # store batch data and update observation
        b_o.append(o)
        b_a.append(a)
        b_r.append(r)
        b_o_.append(o_)
        b_d.append(done)
        b_logp.append(logp)
        if n % timesteps_per_batch == 0:
            yield tuple([
                np.asarray([x[i] for x in b_o]),
                np.asarray([x[i] for x in b_a]),
                np.asarray([x[i] for x in b_r]),
                np.asarray([x[i] for x in b_o_]),
                np.asarray([x[i] for x in b_d]),
                np.asarray([x[i] for x in b_logp]),
            ] for i in range(nenv)) + (infos, )
            for l in (b_o, b_a, b_r, b_o_, b_d, b_logp, infos):
                l.clear()
        o = o_


def _q_retrace(b_r, b_d, b_q_i, b_v_, rho_i, nstep, gamma):
    # size (nenv, nstep, *)
    rho_bar = rho_i.clamp(None, 1)
    qret = b_v_[:, -1]
    qrets = []
    for i in range(nstep - 1, -1, -1):
        qret = b_r[:, i] + gamma * qret * (1 - b_d[:, i])
        qrets.append(qret)
        qret = (rho_bar[:, i] * (qret - b_q_i[:, i])) + b_v_[:, i-1]
    return torch.stack(qrets[::-1], 0)
