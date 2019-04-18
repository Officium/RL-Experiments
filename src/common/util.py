""" Some utils
Note that this file is an MPI-free version of
`https://github.com/openai/baselines/blob/master/baselines/common/*_util.py`
"""
import copy
import random
import re
from importlib import import_module

import gym
import gym.wrappers
import numpy as np
import torch

from common.wrappers import *


# env_id -> env_type
id2type = dict()
for _env in gym.envs.registry.all():
    id2type[_env.id] = _env._entry_point.split(':')[0].rsplit('.', 1)[1]


def build_env(env_id, algorithm, seed, env_type, **kwargs):
    """ Build env based on options """
    if env_type == 'atari':
        env = make_atari(env_id)
        env = wrap_deepmind(env, frame_stack=True)
    else:
        env = gym.make(env_id)
        if algorithm != 'her' and \
                isinstance(env.observation_space, gym.spaces.Dict):
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
    env = RewardScaler(env, kwargs.get('reward_scale', 1))
    env.seed(seed)
    return env


def get_algorithm_defaults(env_type, algorithm):
    """ Get algorithm default hyper-parameters by env """
    try:
        module = get_algorithm_module(algorithm, 'default')
        kwargs = getattr(module, env_type)()
    except (ImportError, AttributeError):
        kwargs = dict()
    return kwargs


def get_algorithm_module(algorithm, submodule):
    """ Get algorithm module in the corresponding folder """
    return import_module('.'.join(['baselines', algorithm, submodule]))


def learn(env_id, algorithm, seed, **kwargs):
    """ Learn entry """
    env_type = id2type[env_id]
    env = build_env(env_id, algorithm, seed, env_type, **kwargs)
    algorithm = algorithm.lower()
    specific_options = get_algorithm_defaults(env_type, algorithm)
    for k, v in kwargs.items():
        if v is not None:
            specific_options[k] = v
    module = get_algorithm_module(algorithm, algorithm)
    return getattr(module, 'learn')(env=env, seed=seed, **specific_options)


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
    def __init__(self, *record_keys):
        assert set(record_keys).issubset({
            'o', 'a', 'r', 'done', 'logp', 'p', 'vpred'
        })
        self._keys = record_keys
        self._records = dict()
        for key in record_keys:
            self._records[key] = []
        self._records['loginfo'] = None

    def append(self, *records):
        assert len(records) == len(self._keys)
        for key, record in zip(self._keys, records):
            if key == 'done':
                record = int(record)
            self._records[key].append(record)

    def __len__(self):
        return len(self._records[self._keys[0]])

    def export(self, keys, device, gamma,
               next_is_done, next_value=None, gae_lam=1.0):
        d = self._records['done']
        r = self._records['r']
        e_reward = sum(r) / sum(d)
        n = len(self._records['r'])
        if 'vpred' in self._keys:
            v = self._records['vpred']
            gae = np.empty(n)
            last_gae = 0
            for i in reversed(range(n)):
                if i == n - 1:
                    flag = 1 - next_is_done
                else:
                    flag = 1 - d[i + 1]
                    next_value = v[i + 1]
                delta = r[i] + gamma * next_value * flag - v[i]
                gae[i] = last_gae = delta + gamma * gae_lam * flag * last_gae
            for i in range(n):
                self._records['r'][i] = gae[i] + v[i]
        else:
            for i in reversed(range(n - 1)):
                self._records['r'][i] += gamma * (0 if d[i] else r[i + 1])

        res = [{'e_reward': e_reward}]
        for key in keys:
            tensor = torch.Tensor(self._records[key]).to(device)
            if key == 'o':
                tensor = tensor.float()
            elif key == 'a':
                if isinstance(self._records[key][0], np.int64):
                    tensor = tensor.long().unsqueeze(1)
                else:
                    tensor = tensor.float()  # continuous case
            elif key in {'r', 'done', 'logp', 'p', 'vpred'}:
                tensor = tensor.float().unsqueeze(1)
            res.append(tensor)

        for key in self._keys:
            self._records[key].clear()

        return tuple(res)
