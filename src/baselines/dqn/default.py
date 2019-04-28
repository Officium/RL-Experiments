import torch.nn as nn
from torch.optim import Adam

from common.util import Flatten


def atari(env):
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n
    network = CNN(in_dim, policy_dim, True)
    optimizer = Adam(network.parameters(), 1e-4, eps=1e-5)
    return dict(
        network=network,
        optimizer=optimizer,
        grad_norm=10,
        batch_size=32,
        double_q=True,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        param_noise=False,
        ob_scale=1 / 255.0
    )


def classic_control(env):
    in_dim = env.observation_space.shape[0]
    policy_dim = env.action_space.n
    network = MLP(in_dim, policy_dim, True)
    optimizer = Adam(network.parameters(), 1e-2, eps=1e-5)
    return dict(
        network=network,
        optimizer=optimizer,
        grad_norm=10,
        batch_size=100,
        double_q=True,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=200,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        param_noise=False,
        ob_scale=1
    )


class CNN(nn.Module):
    def __init__(self, in_shape, out_dim, dueling):
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
        )

        self.q = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, out_dim)
        )
        if dueling:
            self.state = nn.Sequential(
                nn.Linear(cnn_out_dim, 256),
                nn.ReLU(True),
                nn.Linear(256, 1)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        qvalue = self.q(latent)
        if hasattr(self, 'state'):
            qvalue = self.state(latent) + qvalue - qvalue.mean(1, keepdim=True)
        return qvalue


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dueling):
        super().__init__()
        self.feature = nn.Sequential(
            Flatten(),
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.q = nn.Linear(64, out_dim)
        if dueling:
            self.state = nn.Linear(64, 1)

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        qvalue = self.q(latent)
        if hasattr(self, 'state'):
            qvalue = self.state(latent) + qvalue - qvalue.mean(1, keepdim=True)
        return qvalue
