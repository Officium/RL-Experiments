import os
import time
from collections import deque

import torch
import torch.distributions

from common.util import scale_ob, Trajectories


def learn(logger,
          device,
          env, nenv,
          number_timesteps,
          network, optimizer,
          save_path, save_interval, ob_scale,
          gamma, timesteps_per_batch):
    """
    Paper:
    Williams R J. Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning[J]. Machine Learning, 1992: 229-256.

    Parameters:
    ----------
        gamma (float): reward gamma
        batch_episode (int): how many episodes will be sampled before update

    """
    policy = network.to(device)
    generator = _generate(device, env, policy, ob_scale,
                          number_timesteps // nenv, gamma, timesteps_per_batch)

    n_iter = 0
    total_timesteps = 0
    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}
    start_ts = time.time()
    while True:
        try:
            batch = generator.__next__()
        except StopIteration:
            break
        b_o, b_a, b_r, info = batch
        total_timesteps += b_o.size(0)
        for d in info:
            infos['eplenmean'].append(d['l'])
            infos['eprewmean'].append(d['r'])

        b_logits = policy(b_o)
        dist = torch.distributions.Categorical(logits=b_logits)
        b_logp = dist.log_prob(b_a)
        loss = -(b_logp * b_r).sum()  # likelihood ratio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter += 1
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
    record = ['o', 'a', 'r', 'done']
    export = ['o', 'a', 'r']
    trajectories = Trajectories(record, export, device, gamma, ob_scale)

    o = env.reset()
    infos = []
    for n in range(1, number_timesteps + 1):
        # sample action
        with torch.no_grad():
            logits = policy(scale_ob(o, device, ob_scale))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample().cpu().numpy()

        # take action in env
        o_, r, done, info = env.step(a)
        for d in info:
            if d.get('episode'):
                infos.append(d['episode'])

        # store batch data and update observation
        trajectories.append(o, a, r, done)
        if n % timesteps_per_batch == 0:
            yield trajectories.export() + (infos, )
            infos.clear()
        o = o_
