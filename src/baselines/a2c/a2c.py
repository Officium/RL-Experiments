import os
import time
from collections import deque

import torch
import torch.distributions
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from common.util import scale_ob, Trajectories


def learn(logger,
          device,
          env, nenv,
          number_timesteps,
          network, optimizer,
          save_path, save_interval, ob_scale,
          gamma, grad_norm, timesteps_per_batch, ent_coef, vf_coef):
    """
    Paper:
    Mnih V, Badia A P, Mirza M, et al. Asynchronous methods for deep
    reinforcement learning[C]// International Conference on Machine Learning.
    2016: 1928-1937.

    Parameters:
    ----------
    gram_norm (float | None): grad norm
    timesteps_per_batch (int): number of steps per update
    ent_coef (float): policy entropy coefficient in the objective
    vf_coef (float): value function loss coefficient in the objective

    """
    policy = network.to(device)
    number_timesteps = number_timesteps // nenv
    generator = _generate(
        device, env, policy, ob_scale,
        number_timesteps, gamma, timesteps_per_batch
    )
    max_iter = number_timesteps // timesteps_per_batch
    scheduler = LambdaLR(optimizer, lambda i_iter: 1 - i_iter / max_iter)

    total_timesteps = 0
    infos = {k: deque(maxlen=100)
             for k in ['eplenmean', 'eprewmean', 'pgloss', 'v', 'entropy']}
    start_ts = time.time()
    for n_iter in range(1, max_iter + 1):
        scheduler.step()

        batch = generator.__next__()
        b_o, b_a, b_r, b_v_old, info = batch
        for d in info:
            infos['eplenmean'].append(d['l'])
            infos['eprewmean'].append(d['r'])
        total_timesteps += b_o.size(0)

        # calculate advantange
        b_logits, b_v = policy(b_o)
        b_v = b_v[:, 0]
        dist = torch.distributions.Categorical(logits=b_logits)
        entropy = dist.entropy().mean()
        b_logp = dist.log_prob(b_a)
        adv = b_r - b_v_old

        # update policy
        vloss = (b_v - b_r).pow(2).mean()
        pgloss = -(adv * b_logp).mean()
        loss = pgloss + vf_coef * vloss - ent_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        if grad_norm is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), grad_norm)
        optimizer.step()

        # record logs
        infos['pgloss'].append(pgloss.item())
        infos['v'].append(vloss.item())
        infos['entropy'].append(entropy.item())
        logger.info('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
        fps = int(total_timesteps / (time.time() - start_ts))
        logger.info('Total timesteps {} FPS {}'.format(total_timesteps, fps))
        for k, v in infos.items():
            v = (sum(v) / len(v)) if v else float('nan')
            logger.info('{}: {:.6f}'.format(k, v))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.checkpoint'.format(n_iter)))


def _generate(device, env, policy, ob_scale,
              number_timesteps, gamma, timesteps_per_batch):
    """ Generate trajectories """
    record = ['o', 'a', 'r', 'done', 'vpred']
    export = ['o', 'a', 'r', 'vpred']
    trajectories = Trajectories(record, export, device, gamma, ob_scale)

    o = env.reset()
    infos = []
    for n in range(1, number_timesteps + 1):
        # sample action
        with torch.no_grad():
            logits, v = policy(scale_ob(o, device, ob_scale))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample().cpu().numpy()
            v = v.cpu().numpy()[:, 0]

        # take action in env
        o_, r, done, info = env.step(a)
        for d in info:
            if d.get('episode'):
                infos.append(d['episode'])

        # store batch data and update observation
        trajectories.append(o, a, r, done, v)
        if n % timesteps_per_batch == 0:
            with torch.no_grad():
                ob = scale_ob(o_, device, ob_scale)
                v_ = policy(ob)[1].cpu().numpy()[:, 0] * (1 - done)
            yield trajectories.export(v_) + (infos, )
            infos.clear()
        o = o_
