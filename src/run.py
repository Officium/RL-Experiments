""" Run script """
import argparse
import os

import torch

from common.util import learn, parse_all_args


""" Some notice """
print("""
    Notes:
        Log will be recorded at ../logs
        CUDA usage is depend on `CUDA_VISIABLE_DEVICES`
        Modifiy common/models.py to specify parameters of network and optimizer
""")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


""" Parse arguments """
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--env', type=str, required=True, help='environment ID')
parser.add_argument('--algorithm', type=str, required=True, help='Algorithm')
parser.add_argument('--nenv', type=int, default=0, help='parrallel number')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--number_timesteps', type=float, default=1e6)
parser.add_argument('--network', default=None,
                    help='mlp|cnn|lstm|cnnlstm|smallcnn')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--reward_scale', type=float, default=1.0)
parser.add_argument('--save_path', type=str, default='../checkpoints')
parser.add_argument('--save_interval', type=int, default=0,
                    help='Save video and model every x steps (0 = disabled)')
common_options, other_options = parse_all_args(parser)


""" Learn """
if common_options.save_interval:
    os.makedirs(common_options.save_path, exist_ok=True)
model = learn(
    device=device,
    env_id=common_options.env,
    nenv=common_options.nenv,
    seed=common_options.seed,
    number_timesteps=int(common_options.number_timesteps),
    save_path=common_options.save_path,
    save_interval=common_options.save_interval,
    algorithm=common_options.algorithm,
    network=common_options.network,
    optimizer=common_options.optimizer,
    reward_scale=common_options.reward_scale,
    **other_options
)
