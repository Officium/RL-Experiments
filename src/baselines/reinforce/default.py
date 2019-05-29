import math

import torch.nn as nn
from torch.optim import Adam


def classic_control(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    policy_dim = env.action_space.n
    network = MLP(in_dim, policy_dim)
    optimizer = Adam(network.parameters(), 0.01, eps=1e-5)
    params = dict(
        network=network,
        optimizer=optimizer,
        gamma=0.99,
        timesteps_per_batch=100,
        ob_scale=1.0,
    )
    params.update(kwargs)
    return params


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

    def forward(self, x):
        latent = self.feature(x)
        return self.policy(latent)
