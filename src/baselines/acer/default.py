import math

import torch.nn as nn
from torch.optim import RMSprop

from common.util import Flatten


def atari(env):
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n
    network = CNN(in_dim, policy_dim)
    optimizer = RMSprop(network.parameters(), 7e-4, eps=1e-5)
    return dict(
        network=network,
        optimizer=optimizer,
        timesteps_per_batch=20,
        vf_coef=0.5,
        ent_coef=0.01,
        grad_norm=10,
        gamma=0.99,
        buffer_size=50000,
        replay_ratio=4,
        learning_starts=10000,
        c=10.0,
        trust_region=True,
        alpha=0.99,
        max_kl=1
    )


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

        self.policy = nn.Sequential(
            nn.Linear(512, policy_dim),
            nn.Softmax(1)
        )
        nn.init.orthogonal_(self.policy[0].weight, 1e-2)
        nn.init.constant_(self.policy[0].bias, 0)

        self.q = nn.Linear(512, policy_dim)
        nn.init.orthogonal_(self.q.weight, 1)
        nn.init.constant_(self.q.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        prob, q = self.policy(latent), self.q(latent)
        v = (prob * q).sum(-1, keepdim=True)
        return prob, q, v
