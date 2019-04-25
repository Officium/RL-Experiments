""" Build policy/value/... networks and optimizers """
import math

import torch.nn as nn
from torch.optim import Adam, RMSprop


def build_policy(env, name, estimate_value=False, estimate_q=False):
    name = name.upper()
    in_dim = env.observation_space.shape
    if name == 'MLP':
        in_dim = flatten_dim(in_dim)
        policy_dim = env.action_space.n
        value_dim = policy_dim if estimate_q else int(estimate_value)
        return MLP(in_dim, policy_dim, value_dim)
    elif name == 'SMALLCNN':
        policy_dim = env.action_space.n
        value_dim = policy_dim if estimate_q else int(estimate_value)
        return SMALLCNN(in_dim, policy_dim, value_dim)
    elif name == 'CNN':
        policy_dim = env.action_space.n
        value_dim = policy_dim if estimate_q else int(estimate_value)
        return CNN(in_dim, policy_dim, value_dim)
    else:
        raise NotImplementedError


def build_value(env, name, estimate_q=False):
    name = name.upper()
    out_dim = env.action_space.n if estimate_q else 1
    in_dim = env.observation_space.shape
    if name == 'MLP':
        return MLP(flatten_dim(in_dim), 0, out_dim)
    elif name == 'SMALLCNN':
        return SMALLCNN(in_dim, 0, out_dim)
    elif name == 'CNN':
        return CNN(in_dim, 0, out_dim)
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, in_dim, policy_dim, value_dim):
        super().__init__()
        self.feature = nn.Sequential(
            Flatten(),
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0)

        if policy_dim:
            self.policy = nn.Linear(64, policy_dim)
            nn.init.orthogonal_(self.policy.weight, 1e-2)
            nn.init.constant_(self.policy.bias, 0)

        if value_dim:
            self.value = nn.Linear(64, value_dim)
            nn.init.orthogonal_(self.value.weight, 1)
            nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        if hasattr(self, 'policy'):
            if hasattr(self, 'value'):
                return self.policy(latent), self.value(latent)
            else:
                return self.policy(latent)
        else:
            return self.value(latent)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class SMALLCNN(nn.Module):
    def __init__(self, in_shape, policy_dim, value_dim):
        super().__init__()
        c, h, w = in_shape
        cnn_out_dim = 16 * ((h - 12) // 8) * ((w - 12) // 8)
        self.feature = nn.Sequential(
            nn.Conv2d(c, 8, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 4, 2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(True)
        )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0)

        if policy_dim:
            self.policy = nn.Linear(128, policy_dim)
            nn.init.orthogonal_(self.policy.weight, 1e-2)
            nn.init.constant_(self.policy.bias, 0)

        if value_dim:
            self.value = nn.Linear(128, value_dim)
            nn.init.orthogonal_(self.value.weight, 1)
            nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        if hasattr(self, 'policy'):
            if hasattr(self, 'value'):
                return self.policy(latent), self.value(latent)
            else:
                return self.policy(latent)
        else:
            return self.value(latent)


class CNN(nn.Module):
    def __init__(self, in_shape, policy_dim, value_dim):
        super().__init__()
        c, h, w = in_shape
        cnn_out_dim = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.feature = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(True)
        )
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0)

        if policy_dim:
            self.policy = nn.Linear(512, policy_dim)
            nn.init.orthogonal_(self.policy.weight, 1e-2)
            nn.init.constant_(self.policy.bias, 0)

        if value_dim:
            self.value = nn.Linear(512, value_dim)
            nn.init.orthogonal_(self.value.weight, 1)
            nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        if hasattr(self, 'policy'):
            if hasattr(self, 'value'):
                return self.policy(latent), self.value(latent)
            else:
                return self.policy(latent)
        else:
            return self.value(latent)


def get_optimizer(name, parameters, lr):
    """ Get optimizer by name """
    name = name.upper()
    if name == 'ADAM':
        return Adam(parameters, lr, eps=1e-5)
    elif name == 'RMSPROP':
        return RMSprop(parameters, lr, eps=1e-5)
    else:
        raise NotImplementedError


def flatten_dim(size):
    """ Returen flatten dim """
    s = 1
    for x in size:
        s *= x
    return s
