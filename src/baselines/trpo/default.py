import math

import torch.nn as nn
from torch.optim import Adam

from common.util import Flatten


def atari(env, **kwargs):
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n
    policy = SMALLCNN(in_dim, policy_dim)
    value = SMALLCNN(in_dim, 1)
    optimizer = Adam(value.parameters(), 1e-4, eps=1e-5)
    params = dict(
        network=(policy, value),
        optimizer=optimizer,
        gamma=0.98,
        timesteps_per_batch=512,
        cg_iters=10,
        cg_damping=1e-3,
        max_kl=0.001,
        gae_lam=1.0,
        vf_iters=3,
        entcoeff=0.00,
        ob_scale=1 / 255.0
    )
    params.update(kwargs)
    return params


def classic_control(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    policy_dim = env.action_space.n
    policy = MLP(in_dim, policy_dim)
    value = MLP(in_dim, 1)
    optimizer = Adam(value.parameters(), 1e-2, eps=1e-5)
    params = dict(
        network=(policy, value),
        optimizer=optimizer,
        gamma=0.98,
        timesteps_per_batch=512,
        cg_iters=10,
        cg_damping=1e-3,
        max_kl=0.001,
        gae_lam=1.0,
        vf_iters=3,
        entcoeff=0.00,
        ob_scale=1
    )
    params.update(kwargs)
    return params


class SMALLCNN(nn.Module):
    def __init__(self, in_shape, out_dim):
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

        self.policy = nn.Linear(128, out_dim)
        nn.init.orthogonal_(self.policy.weight, 1e-2)
        nn.init.constant_(self.policy.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        return self.policy(latent)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
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

        self.policy = nn.Linear(64, out_dim)
        nn.init.orthogonal_(self.policy.weight, 1e-2)
        nn.init.constant_(self.policy.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        return self.policy(latent)
