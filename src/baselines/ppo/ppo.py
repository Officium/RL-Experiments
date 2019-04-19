from math import ceil
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
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
    set_global_seeds(env, seed)

    policy = build_policy(env, network, estimate_value=True).to(device)
    optimizer = get_optimizer(optimizer, policy.parameters(), lr)
    generator = _generate(
        device, env, policy,
        number_timesteps, gamma, gae_lam, timesteps_per_batch
    )
    max_iter = number_timesteps // timesteps_per_batch
    scheduler = LambdaLR(optimizer, lambda i_iter: 1 - i_iter / max_iter)

    n_iter = 0
    while True:
        scheduler.step()
        try:
            batch = generator.__next__()
        except StopIteration:
            break

        *data, info = batch
        batch_size = ceil(data[0].size(0) / nminibatches)
        loader = DataLoader(list(zip(*data)), batch_size)
        for _ in range(opt_iter):
            for b_o, b_a, b_r, b_logp_old, b_v_old in loader:
                # calculate advantange
                b_logp, b_v = policy(b_o)
                entropy = -(b_logp * b_logp.exp()).sum(-1).mean()
                b_logp = b_logp.gather(1, b_a)
                advantage = b_r - b_v
                # highlight: this normalization gives better performance
                advantage = (advantage - advantage.mean()) / advantage.std()

                # update policy
                c_b_v = b_v_old + (b_v - b_v_old).clamp(-cliprange, cliprange)
                vloss = 0.5 * torch.mean(torch.max(
                    (b_v - b_r).pow(2),
                    (c_b_v - b_r).pow(2)
                ))  # highlight: Clip is also applied to value loss
                ratio = (b_logp - b_logp_old).exp()
                pgloss = torch.mean(torch.max(
                    -advantage * ratio,
                    -advantage * ratio.clamp(1 - cliprange, 1 + cliprange)
                ))
                loss = pgloss + vf_coef * vloss - ent_coef * entropy
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
    record = ['o', 'a', 'r', 'logp', 'vpred', 'done']
    export = ['o', 'a', 'r', 'logp', 'vpred']
    trajectories = Trajectories(record, export, device, gamma, gae_lam)

    o = env.reset()
    for n in range(number_timesteps):
        # sample action
        with torch.no_grad():
            logp, v = policy(torch.Tensor(o).unsqueeze(0).to(device))
            a = logp.exp().multinomial(1).cpu().numpy()[0, 0]
            logp = logp.cpu().numpy()[0, a]
            v = v.cpu().numpy()[0, 0]

        # take action in env
        o_, r, done, info = env.step(a)

        # store batch data and update observation
        if (len(trajectories) + 1) % timesteps_per_batch == 0:
            trajectories.append(o, a, r, logp, v, True)
            with torch.no_grad():
                if done:
                    next_v = 0
                else:
                    _, next_v = policy(torch.Tensor(o_).unsqueeze(0).to(device))
                    next_v = next_v.cpu().numpy()[0, 0]
            yield trajectories.export(done, next_v)
        else:
            trajectories.append(o, a, r, logp, v, True)
        o = env.reset() if done else o_
