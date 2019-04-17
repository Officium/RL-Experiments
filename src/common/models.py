""" Build policy/value/... networks and optimizers """
import torch.nn as nn
from torch.optim import *


def build_policy(env, name):
    name = name.upper()
    if name == 'MLP':
        input_dim = 1
        for n in env.observation_space.shape:
            input_dim *= n
        output_dim = env.action_space.n
        return MLP(input_dim, output_dim)
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_hidden=64):
        super().__init__()
        modules = []
        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Linear(input_dim, num_hidden))
                modules.append(nn.ReLU(True))
            elif i == num_layers - 1:
                modules.append(nn.Linear(num_hidden, output_dim))
                modules.append(nn.LogSoftmax(1))
            else:
                modules.append(nn.Linear(num_hidden, num_hidden))
                modules.append(nn.ReLU(True))
        self.feature = nn.Sequential(*modules)

    def forward(self, x):
        return self.feature(x.contiguous().view(x.size(0), -1))


def get_optimizer(name, parameters, lr):
    """ Get optimizer by name """
    name = name.upper()
    if name == 'ADAM':
        return Adam(parameters, lr)
    else:
        raise NotImplementedError
