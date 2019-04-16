""" Run script """
import argparse
import re

from common.env_utils import build_env
from common.learn_utils import get_algorithm_defaults, learn


""" Parse arguments """
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--env', type=str, default='Pong', help='environment ID')
parser.add_argument('--env_type', type=str, required=False,
                    help='used if the environment type can\'t be auto detected')
parser.add_argument('--seed', type=int, default=None, help='random seed', )
parser.add_argument('--algorithm', type=str, default='PPO', help='Algorithm')
parser.add_argument('--number_timesteps', type=float, default=1e6)
parser.add_argument('--network', default=None,
                    help='one of (mlp, cnn, lstm, cnn_lstm, conv_only)')
parser.add_argument('--reward_scale', type=float, default=1.0,
                    help='Reward scale factor')
parser.add_argument('--save_path', default=None, type=str)
parser.add_argument('--save_interval', default=0, type=int,
                    help='Save video and model every x steps (0 = disabled)')
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

""" Prepare learning """
total_timesteps = int(common_options.num_timesteps)
seed = common_options.seed
algorithm_options = get_algorithm_defaults(common_options)
algorithm_options.update(other_options)
env = build_env(common_options)

""" Learn """
model = learn(
    env=env,
    seed=seed,
    total_timesteps=total_timesteps,
    save_path=common_options.save_path,
    save_internal=common_options.save_internal,
    **algorithm_options
)
