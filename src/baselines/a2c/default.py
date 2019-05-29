import math

import torch.nn as nn
from torch.optim import Adam

from common.util import Flatten


def atari(env, **kwargs):
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n
    network = CNN(in_dim, policy_dim)
    optimizer = Adam(network.parameters(), 7e-4, eps=1e-5)
    params = dict(
        network=network,
        optimizer=optimizer,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=5,
        ent_coef=.01,
        vf_coef=0.5,
        ob_scale=1.0 / 255
    )
    params.update(kwargs)
    return params


def classic_control(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    policy_dim = env.action_space.n
    network = MLP(in_dim, policy_dim)
    optimizer = Adam(network.parameters(), 1e-2, eps=1e-5)
    params = dict(
        network=network,
        optimizer=optimizer,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=5,
        ent_coef=0,
        vf_coef=0.5,
        ob_scale=1
    )
    params.update(kwargs)
    return params


class CNN(nn.Module):
    def __init__(self, in_shape, policy_dim):
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

        self.policy = nn.Linear(512, policy_dim)
        nn.init.orthogonal_(self.policy.weight, 1e-2)
        nn.init.constant_(self.policy.bias, 0)

        self.value = nn.Linear(512, 1)
        nn.init.orthogonal_(self.value.weight, 1)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        return self.policy(latent), self.value(latent)


class MLP(nn.Module):
    def __init__(self, in_dim, policy_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0)

        self.policy = nn.Linear(64, policy_dim)
        nn.init.orthogonal_(self.policy.weight, 1e-2)
        nn.init.constant_(self.policy.bias, 0)

        self.value = nn.Linear(64, 1)
        nn.init.orthogonal_(self.value.weight, 1)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        return self.policy(latent), self.value(latent)
