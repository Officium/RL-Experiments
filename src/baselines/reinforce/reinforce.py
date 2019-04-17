import os

import torch
from torch.utils.data import DataLoader

from common.logger import get_logger
from common.models import build_policy, get_optimizer


def learn(device,
          env, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval,
          gamma, lr, batch_episode, batch_size, **kwargs):
    """
    Paper:
    Williams R J. Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning[J]. Machine Learning, 1992: 229-256.

    Parameters:
    ----------
        gamma (float): reward gamma
        lr (float): learning rate
        batch_episode (int): how many episodes will be sampled before update
        batch_size (int): minibatch size of update
    """
    name = '{}_{}'.format(os.path.split(__file__)[-1][:-3], seed)
    logger = get_logger(name)

    policy = build_policy(env, network).to(device)
    optimizer = get_optimizer(optimizer, policy.parameters(), lr)

    timesteps = 0
    n_iter = 0
    while timesteps < number_timesteps:
        s = env.reset()
        trajectories = []  # s, a, r
        reward = 0
        for _ in range(batch_episode):
            done = False
            n = 0  # length of episode
            while not done:
                with torch.no_grad():
                    logp = policy(torch.Tensor(s).unsqueeze(0).to(device))
                    a = logp.exp().multinomial(1).cpu().numpy()[0, 0]
                s_, r, done, info = env.step(a)
                trajectories.append([s, a, r])
                n += 1
                s = s_
            for t in range(1, n):
                trajectories[-t - 1][2] += trajectories[-t][2] * gamma
            reward += trajectories[-n][2]
        reward /= batch_episode

        loader = DataLoader(trajectories, batch_size, shuffle=True)
        for b_s, b_a, b_r in loader:
            b_s = b_s.float().to(device)
            b_a = b_a.long().unsqueeze(1).to(device)
            b_r = b_r.float().unsqueeze(1).to(device)
            b_logp = policy(b_s).gather(1, b_a)
            loss = -(b_logp * b_r).sum()  # likelihood ratio
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timesteps += len(trajectories)
        n_iter += 1
        logger.info('Iter {}, Pass timestamps {} Reward {:.2f}'
                    .format(n_iter, timesteps, reward))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.{}'.format(name, n_iter)))
