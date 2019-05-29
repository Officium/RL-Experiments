import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim import Adam

from common.util import Flatten


def atari(env, **kwargs):
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n
    params = dict(
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
        dueling=True,
        atom_num=51,
        min_value=-10,
        max_value=10,
        ob_scale=1 / 255.0
    )
    params.update(kwargs)
    network = CNN(in_dim, policy_dim, params['atom_num'], params.pop('dueling'))
    optimizer = Adam(network.parameters(), 1e-4, eps=1e-5)
    params.update(network=network, optimizer=optimizer)
    return params


def classic_control(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    policy_dim = env.action_space.n
    params = dict(
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
        dueling=True,
        atom_num=1,
        min_value=-10,
        max_value=10,
        ob_scale=1
    )
    params.update(kwargs)
    network = MLP(in_dim, policy_dim, params['atom_num'], params.pop('dueling'))
    optimizer = Adam(network.parameters(), 1e-3, eps=1e-5)
    params.update(network=network, optimizer=optimizer)
    return params


class CNN(nn.Module):
    def __init__(self, in_shape, out_dim, atom_num, dueling):
        super().__init__()
        c, h, w = in_shape
        cnn_out_dim = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.atom_num = atom_num
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
            nn.Linear(256, out_dim * atom_num)
        )
        if dueling:
            self.state = nn.Sequential(
                nn.Linear(cnn_out_dim, 256),
                nn.ReLU(True),
                nn.Linear(256, atom_num)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        latent = self.feature(x)
        qvalue = self.q(latent)
        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            qvalue = qvalue.view(batch_size, -1, self.atom_num)
            if hasattr(self, 'state'):
                svalue = self.state(latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, atom_num, dueling):
        super().__init__()
        self.atom_num = atom_num
        self.feature = nn.Sequential(
            Flatten(),
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.q = nn.Linear(64, out_dim * atom_num)
        if dueling:
            self.state = nn.Linear(64, atom_num)

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        latent = self.feature(x)
        qvalue = self.q(latent)
        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            if hasattr(self, 'state'):
                qvalue = qvalue.view(batch_size, -1, self.atom_num)
                svalue = self.state(latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs
