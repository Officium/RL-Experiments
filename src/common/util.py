""" Some utils
Note that this file is an MPI-free version of
`https://github.com/openai/baselines/blob/master/baselines/common/*_util.py`
"""
import re
from importlib import import_module

import gym
import gym.wrappers

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
    float_pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    while other_args:  # only str, int and float type will be parsed
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
