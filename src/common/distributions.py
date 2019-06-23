"""Wrappers for policy distributions"""
import torch
import torch.distributions


class Categorical(object):
    """Categorical policy distribution"""
    def __init__(self, logits):
        self._dist = torch.distributions.Categorical(logits=logits)

    def entropy(self):
        return self._dist.entropy()

    def log_prob(self, b_a):
        return self._dist.log_prob(b_a.long())

    def sample(self):
        return self._dist.sample()

    @classmethod
    def __repr__(cls):
        return 'Categorical'


class DiagGaussian(object):
    """DiagGaussian policy distribution"""
    def __init__(self, mean_logstd):
        size = mean_logstd.size(1) // 2
        mean, logstd = torch.split(mean_logstd, size, 1)
        self._dist = torch.distributions.Normal(mean, logstd.exp())

    def entropy(self):
        return self._dist.entropy().sum(-1)

    def log_prob(self, b_a):
        return self._dist.log_prob(b_a.float()).sum(-1)

    def sample(self):
        return self._dist.sample()

    @classmethod
    def __repr__(cls):
        return 'DiagGaussian'
