""" Build policy/value/... networks and optimizers """
import torch
import torch.nn as nn
from torch.optim import *


def build_policy(env, name, estimate_value=False, estimate_q=False):
    name = name.upper()
    in_dim = env.reset().shape
    if name == 'MLP':
        in_dim = flatten_dim(in_dim)
        policy_dim = env.action_space.n
        value_dim = policy_dim if estimate_q else int(estimate_value)
        return MLP(in_dim, policy_dim, value_dim)
    elif name == 'SMALLCNN':
        policy_dim = env.action_space.n
        value_dim = policy_dim if estimate_q else int(estimate_value)
        return SMALLCNN(in_dim, policy_dim, value_dim)
    else:
        raise NotImplementedError


def build_value(env, name, estimate_q=False):
    name = name.upper()
    out_dim = env.action_space.n if estimate_q else 1
    in_dim = env.reset().shape
    if name == 'MLP':
        return MLP(flatten_dim(in_dim), 0, out_dim)
    elif name == 'SMALLCNN':
        return SMALLCNN(in_dim, 0, out_dim)
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, in_dim, policy_dim, value_dim):
        super().__init__()
        self.feature = nn.Sequential(
            Flaten(),
            nn.Linear(in_dim, 64),
            nn.ReLU(True)
        )

        if policy_dim:
            self.policy = nn.Sequential(
                nn.Linear(64, policy_dim),
                nn.LogSoftmax(1)
            )

        if value_dim:
            self.value = nn.Linear(64, value_dim)

    def forward(self, x):
        latent = self.feature(x)
        if hasattr(self, 'policy'):
            logprob = self.policy(latent)
        if hasattr(self, 'value'):
            value = self.value(latent)
        if hasattr(self, 'policy'):
            if hasattr(self, 'value'):
                return logprob, value
            else:
                return logprob
        else:
            return value


class Flaten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class SMALLCNN(nn.Module):
    def __init__(self, in_shape, policy_dim, value_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_shape[0], 8, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 4, 2),
            nn.ReLU(True),
            Flaten()
        )
        dim = self.feature(torch.rand(1, *in_shape)).size(1)
        self.out = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(True)
        )

        if policy_dim:
            self.policy = nn.Sequential(
                nn.Linear(128, policy_dim),
                nn.LogSoftmax(1)
            )

        if value_dim:
            self.value = nn.Linear(128, value_dim)

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        latent = self.out(self.feature(x / 255.0))
        if hasattr(self, 'policy'):
            logprob = self.policy(latent)
        if hasattr(self, 'value'):
            value = self.value(latent)
        if hasattr(self, 'policy'):
            if hasattr(self, 'value'):
                return logprob, value
            else:
                return logprob
        else:
            return value


def get_optimizer(name, parameters, lr):
    """ Get optimizer by name """
    name = name.upper()
    if name == 'ADAM':
        return Adam(parameters, lr)
    else:
        raise NotImplementedError


def flatten_dim(size):
    """ Returen flatten dim """
    s = 1
    for x in size:
        s *= x
    return s
