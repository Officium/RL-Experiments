# -*- coding: utf-8 -*-
""" This file defines the base classes """
import abc
import random


class Agent(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, state, step, noise):
        """
        Args:
            state: State vector.
            step: Time step.
            noise: A D-dimensional noise vector.
        Returns:
            A D dimensional action vector.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Args:
            size: Max number of transitions to store. If the buffer overflows, the old memories would be dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    @property
    def size(self):
        return self._maxsize

    def __len__(self):
        return len(self._storage)

    def add(self, item):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            self._storage[self._next_idx] = item
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Args:
            batch_size: How many transitions to sample.
        """
        n = len(self._storage[0])
        res = tuple(([] for _ in xrange(n)))
        for _ in xrange(batch_size):
            sample = random.choice(self._storage)
            for i in xrange(n):
                res[i].append(sample[i])
        return res


class NoiseGenerator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate(self):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

