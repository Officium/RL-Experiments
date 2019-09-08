""" Some utils
Note that this file is a MPI-free version of
`https://github.com/openai/baselines/blob/master/baselines/common/*_util.py`
"""
import os
import random
import re
from functools import partial
from importlib import import_module
from multiprocessing import cpu_count
from sys import platform

import gym
import numpy as np
import torch
import torch.nn as nn

from common.logger import init_logger
from common.wrappers import *


def build_env(env_id, algorithm, env_type, seed, log_path, **kwargs):
    """ Build env based on options """
    assert env_type in {'atari', 'classic_control', 'box2d'}
    reward_scale = kwargs.pop('reward_scale')
    nenv = kwargs.pop('nenv') or cpu_count() // (1 + (platform == 'darwin'))
    stack = env_type == 'atari'
    if algorithm == 'dqn':
        env = make_env(env_id, env_type, seed, reward_scale, log_path, stack)
    else:
        if algorithm == 'trpo':
            nenv = 1
        kwargs['nenv'] = nenv
        env = make_vec_env(
            env_id, env_type, nenv, seed, reward_scale, log_path, stack)

    return env, kwargs


def make_env(env_id, env_type, seed, reward_scale, log_path, frame_stack=False):
    """ Make env """
    actor_log_path = os.path.join(log_path, 'actor.log.{}'.format(str(seed)))
    if env_type == 'atari':
        env = gym.make(env_id)
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env, actor_log_path)
        # deepmind wrap
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        if frame_stack:
            env = FrameStack(env, 4)
    elif env_type in {'classic_control', 'box2d'}:
        env = Monitor(gym.make(env_id), actor_log_path)
    else:
        raise NotImplementedError
    if reward_scale != 1:
        env = RewardScaler(env, reward_scale)
    env.seed(seed)
    return env


def make_vec_env(env_id, env_type, nenv, seed,
                 reward_scale, log_path, frame_stack=True):
    """ Make vectorized env """
    env = SubprocVecEnv([
        partial(make_env, env_id, env_type, seed + i, reward_scale, log_path)
        for i in range(nenv)
    ])
    if frame_stack:
        env = VecFrameStack(env, 4)
    return env


def get_algorithm_parameters(env, env_type, algorithm, **kwargs):
    """ Get algorithm hyper-parameters """
    module = get_algorithm_module(algorithm, 'default')
    return getattr(module, env_type)(env, **kwargs)


def get_algorithm_module(algorithm, submodule):
    """ Get algorithm module in the corresponding folder """
    return import_module('.'.join(['baselines', algorithm, submodule]))


def learn(env_id, algorithm, env_type, seed, log_path, **kwargs):
    """ Learn entry """
    sub_folder = '{}_{}_{}'.format(env_id, algorithm, seed)
    log_path = os.path.join(log_path, sub_folder)
    logger = init_logger(log_path)

    set_global_seeds(seed)

    env, kwargs = build_env(env_id, algorithm,
                            env_type, seed, log_path, **kwargs)

    algorithm = algorithm.lower()
    options = get_algorithm_parameters(env, env_type, algorithm, **kwargs)
    module = get_algorithm_module(algorithm, algorithm)
    s = '\n' + '-' * 60 + '\n'
    option_repr = ''.join(
        '{}{}: {}'.format(s, k, v.__repr__()) for k, v in options.items()) + s
    logger.info('Start training `{}` on `{}` with settings {}'
                .format(algorithm, env, option_repr))

    try:
        getattr(module, 'learn')(logger, env=env, **options)
    except Exception as e:
        logger.critical('algorithm execute fail', exc_info=e)


def parse_all_args(parser):
    """ Parse known and unknown args """
    common_options, other_args = parser.parse_known_args()
    other_options = dict()
    index = 0
    n = len(other_args)
    float_pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    while index < n:  # only str, int and float type will be parsed
        if other_args[index].startswith('--'):
            if other_args[index].__contains__('='):
                key, value = other_args[index].split('=')
                index += 1
            else:
                key, value = other_args[index:index + 2]
                index += 2
            if re.match(float_pattern, value):
                value = float(value)
                if value.is_integer():
                    value = int(value)
            other_options[key[2:]] = value
    return common_options, other_options


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trajectories(object):
    """
    Parameters:
    -----------
    record_keys (list[str]): stored keys
    export_keys (list[str]): export keys, must be the subset of record_keys
    device (torch.device): export device
    gamma (float): reward gamma
    gae_lam (float): gae lambda

    Note that `done` describe the next timestep
    """
    SUPPORT_KEYS = {'o', 'a', 'r', 'done', 'logp', 'vpred'}

    def __init__(self, record_keys, export_keys,
                 device, gamma, ob_scale, gae_lam=1.0):
        assert set(record_keys).issubset(Trajectories.SUPPORT_KEYS)
        assert set(export_keys).issubset(record_keys)
        assert {'o', 'a', 'r', 'done'}.issubset(set(record_keys))
        self._keys = record_keys
        self._records = dict()
        for key in record_keys:
            self._records[key] = []
        self._export_keys = export_keys
        self._device = device
        self._gamma = gamma
        self._gae_lam = gae_lam
        self._offsets = {'done': [], 'r': []}
        self._ob_scale = ob_scale

    def append(self, *records):
        assert len(records) == len(self._keys)
        for key, record in zip(self._keys, records):
            if key == 'done':
                record = record.astype(int)
                self._offsets['done'].append(record)
            if key == 'r':
                self._offsets['r'].append(record)
            self._records[key].append(record)

    def __len__(self):
        return len(self._records[self._keys[0]])

    def export(self, next_value=None):
        # estimate discounted reward
        d = self._records['done']
        r = self._records['r']
        n = len(self)
        nenv = r[0].shape[0]
        if 'vpred' in self._keys and self._gae_lam != 1:  # gae estimation
            v = self._records['vpred']
            gae = np.empty((nenv, n))
            last_gae = 0
            for i in reversed(range(n)):
                flag = 1 - d[i]
                delta = r[i] + self._gamma * next_value * flag - v[i]
                delta += self._gamma * self._gae_lam * flag * last_gae
                gae[:, i] = last_gae = delta
                next_value = v[i]
            for i in range(n):
                self._records['r'][i] = gae[:, i] + v[i]
        else:  # discount reward
            if next_value is not None:
                self._records['r'][-1] += self._gamma * next_value
            for i in reversed(range(n - 1)):
                self._records['r'][i] += self._gamma * r[i + 1] * (1 - d[i])

        # convert to tensor
        res = []
        for key in self._export_keys:
            shape = (n * nenv, ) + self._records[key][0].shape[1:]
            data = np.asarray(self._records[key], np.float32).reshape(shape)
            tensor = torch.from_numpy(data).to(self._device)
            if key == 'o':
                tensor.mul_(self._ob_scale)
            res.append(tensor)

        for key in self._keys:
            self._records[key].clear()

        return tuple(res)


def scale_ob(array, device, scale):
    return torch.from_numpy(array.astype(np.float32) * scale).to(device)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)
