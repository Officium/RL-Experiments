from math import ceil
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.logger import get_logger
from common.models import build_policy, get_optimizer
from common.util import set_global_seeds, Trajectories


def learn(device,
          env, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval,
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
    set_global_seeds(seed)

    policy = build_policy(env, network, estimate_value=True).to(device)
    optimizer = get_optimizer(optimizer, policy.parameters(), lr)
    generator = _generate(
        device, env, policy,
        number_timesteps, gamma, gae_lam, timesteps_per_batch
    )

    n_iter = 0
    while True:
        try:
            batch = generator.__next__()
        except StopIteration:
            break

        info, batch_o, batch_a, batch_r, batch_logp_old = batch
        batch_size = ceil(batch_o.size(0) / nminibatches)
        zipped_data = list(zip(batch_o, batch_a, batch_r, batch_logp_old))
        loader = DataLoader(zipped_data, batch_size)
        for _ in range(opt_iter):
            for b_o, b_a, b_r, b_logp_old in loader:
                b_logp, b_v = policy(b_o)
                entropy = -(b_logp * b_logp.exp()).sum(-1).mean()
                b_logp = b_logp.gather(1, b_a)
                advantage = b_r - b_v
                advantage = (advantage - advantage.mean()) / advantage.std()

                # update policy
                vloss = (b_v - b_r).pow(2).mean()  # value loss, L^VF
                ratio = (b_logp - b_logp_old).exp()
                clip = torch.min(ratio * advantage,  # L^CLIP
                                 ratio.clamp(1 - cliprange,
                                             1 + cliprange) * advantage).mean()
                loss = -clip + vf_coef * vloss - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                if grad_norm is not None:
                    nn.utils.clip_grad_norm_(policy.parameters(), grad_norm)
                optimizer.step()

        n_iter += 1
        logger.info('Iter {}, Reward {:.2f}'.format(n_iter, info['e_reward']))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.{}'.format(name, n_iter)))


def _generate(device, env, policy,
              number_timesteps, gamma, gae_lam, timesteps_per_batch):
    """ Generate trajectories """
    o = env.reset()
    n = 0
    trajectories = Trajectories('o', 'a', 'r', 'logp', 'vpred', 'done')
    while n < number_timesteps:
        n += 1
        with torch.no_grad():
            logp, v = policy(torch.Tensor(o).unsqueeze(0).to(device))
            a = logp.exp().multinomial(1).cpu().numpy()[0, 0]
            logp = logp.cpu().numpy()[0, a]
            v = v.cpu().numpy()[0, 0]

        o_, r, done, info = env.step(a)

        if (len(trajectories) + 1) % timesteps_per_batch == 0:
            trajectories.append(o, a, r, logp, v, True)
            if not done:
                with torch.no_grad():
                    _, next_v = policy(torch.Tensor(o_).unsqueeze(0).to(device))
                    next_v = next_v.cpu().numpy()[0, 0]
            else:
                next_v = 0
            yield trajectories.export(['o', 'a', 'r', 'logp'],
                                      device, gamma, done, next_v, gae_lam)
        else:
            trajectories.append(o, a, r, logp, v, True)
        if done:
            o = env.reset()
        else:
            o = o_
