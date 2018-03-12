# -*- coding: utf-8 -*-
""" This file defines the base classes """
import abc


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


class Environment(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        """
        Reset the state of the environment.
        """
        raise NotImplementedError("Must be implemented in subclass.")
