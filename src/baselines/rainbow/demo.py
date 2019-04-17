# -*- coding: utf-8 -*-
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import RAINBOW, base


def get_scale_noise(input_size):
    x = torch.randn(input_size)
    return x.sign().mul(x.abs().sqrt())


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.constant_(self.mu.bias, 0)
        self.sigma = nn.Linear(in_features, out_features)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.constant_(self.sigma.bias, 0)
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_noise()

    def forward(self, x):
        weight = self.sigma.weight.data.mul(self.weight_epsilon)
        bias = self.sigma.bias.data.mul(self.bias_epsilon)
        return self.mu(x) + F.linear(x, weight, bias)

    def reset_noise(self):
        weight_epsilon = get_scale_noise(self.out_features).ger(get_scale_noise(self.in_features))
        bias_epsilon = get_scale_noise(self.out_features)
        self.weight_epsilon.copy_(weight_epsilon)
        self.bias_epsilon.copy_(bias_epsilon)


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, atom_num):
        super(Network, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        self.features = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(True),
        )
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.features[0].weight, gain)
        nn.init.constant_(self.features[0].bias, 0)
        self.value = NoisyLinear(64, atom_num)
        self.advantage = NoisyLinear(64, atom_num * action_dim)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.features(x)
        value = self.value(feature).unsqueeze(1)
        advantage = self.advantage(feature).view(batch_size, self.action_dim, self.atom_num)
        feature = value + advantage - advantage.mean(1, keepdim=True)
        logprobs = self.logsoftmax(feature.view(-1, self.atom_num))
        return logprobs.view(batch_size, self.action_dim, self.atom_num)

    def reset_noise(self):
        self.value.reset_noise()
        self.advantage.reset_noise()


torch.random.manual_seed(28)
num_atoms = 21
min_value = -10
max_value = 10
env = gym.make('CartPole-v0')
net = Network(env.observation_space.shape[0], env.action_space.n, num_atoms)
target_net = Network(env.observation_space.shape[0], env.action_space.n, num_atoms)
replay = base.ReplayBuffer(10000)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
agent = RAINBOW.Agent(net, target_net, replay, optimizer, 200, min_value, max_value, num_atoms, 100)
agent.learn(env, 10000, 32)
