import os
import time
from collections import deque
from itertools import chain
from math import ceil

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from common.logger import get_logger
from common.models import build_policy, get_optimizer
from common.util import scale_ob, Trajectories


def learn(device,
          env, nenv, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval, ob_scale,
          lr, gamma, grad_norm, timesteps_per_batch, ent_coef,
          vf_coef, gae_lam, nminibatches, opt_iter, cliprange, **kwargs):
    """
    Paper:
    Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization
    algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.

    Parameters:
    ----------
    gram_norm (float | None): grad norm
    timesteps_per_batch (int): number of steps per update
    ent_coef (float): policy entropy coefficient in the objective
    vf_coef (float): value function loss coefficient in the objective
    gae_lam (float): gae lambda
    nminibatches (int): number of training minibatches per update
    opt_iter (int): number of training iterations per update
    cliprange (float): clipping range

    """
    name = '{}_{}'.format(os.path.split(__file__)[-1][:-3], seed)
    logger = get_logger(name)
    logger.warn('This implementation of ppo only '
                'support discrete action spaces now!')

    policy = build_policy(env, network, estimate_value=True).to(device)
    optimizer = get_optimizer(optimizer, policy.parameters(), lr)
    number_timesteps = number_timesteps // nenv
    generator = _generate(
        device, env, policy, ob_scale,
        number_timesteps, gamma, gae_lam, timesteps_per_batch
    )
    max_iter = number_timesteps // timesteps_per_batch
    scheduler = LambdaLR(optimizer, lambda i_iter: 1 - i_iter / max_iter)

    n_iter = 0
    total_timesteps = 0
    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}
    start_ts = time.time()
    while True:
        scheduler.step()
        try:
            batch = generator.__next__()
        except StopIteration:
            break

        *data, info = batch
        for d in info:
            infos['eplenmean'].append(d['l'])
            infos['eprewmean'].append(d['r'])
        total_timesteps += data[0].size(0)
        batch_size = ceil(data[0].size(0) / nminibatches)
        loader = DataLoader(list(zip(*data)), batch_size, True)
        records = {'pg': [], 'v': [], 'ent': [], 'kl': [], 'clipfrac': []}
        for _ in range(opt_iter):
            for b_o, b_a, b_r, b_logp_old, b_v_old in loader:
                # calculate advantange
                b_logp, b_v = policy(b_o)
                entropy = -(b_logp * b_logp.exp()).sum(-1).mean()
                b_logp = b_logp.gather(1, b_a)
                adv = b_r - b_v
                # highlight: this normalization gives better performance
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # update policy
                c_b_v = b_v_old + (b_v - b_v_old).clamp(-cliprange, cliprange)
                vloss = 0.5 * torch.mean(torch.max(
                    (b_v - b_r).pow(2),
                    (c_b_v - b_r).pow(2)
                ))  # highlight: Clip is also applied to value loss
                ratio = (b_logp - b_logp_old).exp()
                pgloss = torch.mean(torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - cliprange, 1 + cliprange)
                ))
                loss = pgloss + vf_coef * vloss - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                if grad_norm is not None:
                    nn.utils.clip_grad_norm_(policy.parameters(), grad_norm)
                optimizer.step()

                # record logs
                records['pg'].append(pgloss.item())
                records['v'].append(vloss.item())
                records['ent'].append(entropy.item())
                records['kl'].append((b_logp - b_logp_old).pow(2).mean() * 0.5)
                clipfrac = ((ratio - 1).abs() > cliprange).float().mean().item()
                records['clipfrac'].append(clipfrac)

        n_iter += 1
        logger.info('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
        fps = int(total_timesteps / (time.time() - start_ts))
        logger.info('Total timesteps {} FPS {}'.format(total_timesteps, fps))
        for k, v in chain(infos.items(), records.items()):
            v = (sum(v) / len(v)) if v else float('nan')
            logger.info('{}: {:.6f}'.format(k, v))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.{}'.format(name, n_iter)))


def _generate(device, env, policy, ob_scale,
              number_timesteps, gamma, gae_lam, timesteps_per_batch):
    """ Generate trajectories """
    record = ['o', 'a', 'r', 'done', 'logp', 'vpred']
    export = ['o', 'a', 'r', 'logp', 'vpred']
    trajectories = Trajectories(record, export,
                                device, gamma, ob_scale, gae_lam)

    o = env.reset()
    infos = []
    for n in range(1, number_timesteps + 1):
        # sample action
        with torch.no_grad():
            logp, v = policy(scale_ob(o, device, ob_scale))
            a = logp.exp().multinomial(1)
            logp = logp.gather(1, a).cpu().numpy()[:, 0]
            a = a.cpu().numpy()[:, 0]
            v = v.cpu().numpy()[:, 0]

        # take action in env
        o_, r, done, info = env.step(a)
        for d in info:
            if d.get('episode'):
                infos.append(d['episode'])

        # store batch data and update observation
        trajectories.append(o, a, r, done, logp, v)
        if n % timesteps_per_batch == 0:
            with torch.no_grad():
                ob = scale_ob(o_, device, ob_scale)
                v_ = policy(ob)[1].cpu().numpy()[:, 0] * (1 - done)
            yield trajectories.export(v_) + (infos, )
            infos.clear()
        o = o_
