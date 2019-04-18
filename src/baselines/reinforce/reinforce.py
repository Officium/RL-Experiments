import os

import torch

from common.logger import get_logger
from common.models import build_policy, get_optimizer
from common.util import set_global_seeds, Trajectories


def learn(device,
          env, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval,
          gamma, lr, timesteps_per_batch, reset_after_batch, **kwargs):
    """
    Paper:
    Williams R J. Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning[J]. Machine Learning, 1992: 229-256.

    Parameters:
    ----------
        gamma (float): reward gamma
        lr (float): learning rate
        batch_episode (int): how many episodes will be sampled before update
        reset_after_batch (int): whether reset env after batch sample
    """
    name = '{}_{}'.format(os.path.split(__file__)[-1][:-3], seed)
    logger = get_logger(name)
    set_global_seeds(seed)

    policy = build_policy(env, network).to(device)
    optimizer = get_optimizer(optimizer, policy.parameters(), lr)
    generator = _generate(device, env, policy, number_timesteps,
                          gamma, timesteps_per_batch, reset_after_batch)

    n_iter = 0
    while True:
        try:
            batch = generator.__next__()
        except StopIteration:
            break

        info, b_o, b_a, b_r = batch
        b_logp = policy(b_o).gather(1, b_a)
        loss = -(b_logp * b_r).sum()  # likelihood ratio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter += 1
        logger.info('Iter {}, Reward {:.2f}'.format(n_iter, info['e_reward']))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.{}'.format(name, n_iter)))


def _generate(device, env, policy, number_timesteps,
              gamma, timesteps_per_batch, reset_after_batch):
    """ Generate trajectories """
    o = env.reset()
    n = 0
    trajectories = Trajectories('o', 'a', 'r', 'done')
    while n < number_timesteps:
        n += 1
        with torch.no_grad():
            logp = policy(torch.Tensor(o).unsqueeze(0).to(device))
            a = logp.exp().multinomial(1).cpu().numpy()[0, 0]

        o_, r, done, info = env.step(a)

        if (len(trajectories) + 1) % timesteps_per_batch == 0:
            trajectories.append(o, a, r, True)
            yield trajectories.export(['o', 'a', 'r'], device, gamma, done)
            if reset_after_batch or done:
                o = env.reset()
            else:
                o = o_
        else:
            trajectories.append(o, a, r, done)
            if done:
                o = env.reset()
            else:
                o = o_
