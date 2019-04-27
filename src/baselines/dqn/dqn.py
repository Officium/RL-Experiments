import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax

from common.logger import get_logger
from common.models import build_q, get_optimizer
from common.util import scale_ob
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def learn(device,
          env, seed,
          number_timesteps,
          network, optimizer,
          save_path, save_interval, ob_scale,
          lr, gamma, grad_norm,
          double_q, param_noise, dueling,
          exploration_fraction, exploration_final_eps,
          batch_size, train_freq, learning_starts, target_network_update_freq,
          buffer_size, prioritized_replay, prioritized_replay_alpha,
          prioritized_replay_beta0, **kwargs):
    """
    Papers:
    Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep
    reinforcement learning[J]. Nature, 2015, 518(7540): 529.
    Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining Improvements
    in Deep Reinforcement Learning[J]. 2017.

    Parameters:
    ----------
    double_q (bool): if True double DQN will be used
    param_noise (bool): whether or not to use parameter space noise
    dueling (bool): if True dueling value estimation will be used
    exploration_fraction (float): fraction of entire training period over which
                                  the exploration rate is annealed
    exploration_final_eps (float): final value of random action probability
    batch_size (int): size of a batched sampled from replay buffer for training
    train_freq (int): update the model every `train_freq` steps
    learning_starts (int): how many steps of the model to collect transitions
                           for before learning starts
    target_network_update_freq (int): update the target network every
                                      `target_network_update_freq` steps
    buffer_size (int): size of the replay buffer
    prioritized_replay (bool): if True prioritized replay buffer will be used.
    prioritized_replay_alpha (float): alpha parameter for prioritized replay
    prioritized_replay_beta0 (float): beta parameter for prioritized replay

    """
    name = '{}_{}'.format(os.path.split(__file__)[-1][:-3], seed)
    logger = get_logger(name)
    logger.info('Note that Rainbow features supported in current version is '
                'consitent with openai/baselines, which means `Multi-step` and '
                '`Distributional` are missing. Welcome any contributions!')

    qnet = build_q(env, network, dueling).to(device)
    qtar = build_q(env, network, dueling).to(device)
    qtar.load_state_dict(qnet.state_dict())
    optimizer = get_optimizer(optimizer, qnet.parameters(), lr)
    if prioritized_replay:
        buffer = PrioritizedReplayBuffer(buffer_size, device,
                                         prioritized_replay_alpha,
                                         prioritized_replay_beta0)
    else:
        buffer = ReplayBuffer(buffer_size, device)
    generator = _generate(device, env, qnet, ob_scale,
                          number_timesteps, param_noise,
                          exploration_fraction, exploration_final_eps)

    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}
    start_ts = time.time()
    for n_iter in range(1, number_timesteps + 1):
        if prioritized_replay:
            buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
        *data, info = generator.__next__()
        buffer.add(*data)
        for k, v in info.items():
            infos[k].append(v)

        # update qnet
        if n_iter > learning_starts and n_iter % train_freq == 0:
            b_o, b_a, b_r, b_o_, b_d, *extra = buffer.sample(batch_size)
            b_q = qnet(b_o).gather(1, b_a)
            with torch.no_grad():
                if double_q:
                    b_a_ = qnet(b_o_).argmax(1).unsqueeze(1)
                    b_q_ = (1 - b_d) * qtar(b_o_).gather(1, b_a_)
                else:
                    b_q_ = (1 - b_d) * qtar(b_o_).max(1)[0]
            abs_td_error = (b_q - (b_r + gamma * b_q_)).abs()
            if extra:
                loss = (extra[0] * huber_loss(abs_td_error)).mean()  # weighted
            else:
                loss = huber_loss(abs_td_error).mean()
            optimizer.zero_grad()
            loss.backward()
            if grad_norm is not None:
                nn.utils.clip_grad_norm_(qnet.parameters(), grad_norm)
            optimizer.step()
            if prioritized_replay:
                priorities = abs_td_error.detach().cpu().clamp(0)  # stable
                buffer.update_priorities(extra[1], priorities)

        # update target net and log
        if n_iter % target_network_update_freq == 0:
            qtar.load_state_dict(qnet.state_dict())
            logger.info('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            fps = int(n_iter / (time.time() - start_ts))
            logger.info('Total timesteps {} FPS {}'.format(n_iter, fps))
            for k, v in infos.items():
                v = (sum(v) / len(v)) if v else float('nan')
                logger.info('{}: {:.6f}'.format(k, v))

        if save_interval and n_iter % save_interval == 0:
            torch.save([qnet.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.{}'.format(name, n_iter)))


def _generate(device, env, qnet, ob_scale,
              number_timesteps, param_noise,
              exploration_fraction, exploration_final_eps):
    """ Generate training batch sample """
    noise_scale = 1e-2
    action_dim = env.action_space.n
    explore_steps = number_timesteps * exploration_fraction

    o = env.reset()
    infos = dict()
    epret = eplen = 0
    for n in range(1, number_timesteps + 1):
        epsilon = 1.0 - (1.0 - exploration_final_eps) * n / explore_steps
        epsilon = max(exploration_final_eps, epsilon)

        # sample action
        with torch.no_grad():
            ob = scale_ob(np.expand_dims(o, 0), device, ob_scale)
            q = qnet(ob)
            if not param_noise:
                if random.random() < epsilon:
                    a = int(random.random() * action_dim)
                else:
                    a = q.argmax(1).cpu().numpy()[0]
            else:
                # see Appendix C of `https://arxiv.org/abs/1706.01905`
                q_dict = qnet.state_dict()
                for _, m in qnet.named_modules():
                    if isinstance(m, nn.Linear):
                        std = torch.empty_like(m.weight).fill_(noise_scale)
                        m.weight.add_(torch.normal(0, std).to(device))
                        std = torch.empty_like(m.bias).fill_(noise_scale)
                        m.bias.add_(torch.normal(0, std).to(device))
                q_perturb = qnet(ob)
                kl_perturb = (
                    (log_softmax(q) - log_softmax(q_perturb)) * softmax(q)
                ).sum(-1).mean()
                kl_explore = -math.log(1 - epsilon + epsilon / action_dim)
                if kl_perturb < kl_explore:
                    noise_scale *= 1.01
                else:
                    noise_scale /= 1.01
                qnet.load_state_dict(q_dict)
                if random.random() < epsilon:
                    a = int(random.random() * action_dim)
                else:
                    a = q_perturb.argmax(1).cpu().numpy()[0]

        # take action in env
        o_, r, done, _ = env.step(a)
        epret += r
        eplen += 1
        if done:
            infos = {'eprewmean': epret, 'eplenmean': eplen}
            epret = eplen = 0

        # return data and update observation
        yield (o, a, r, o_, int(done), infos)
        infos = dict()
        o = o_ if not done else env.reset()


def huber_loss(abs_td_error):
    flag = (abs_td_error < 1).float()
    return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

