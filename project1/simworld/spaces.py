from abc import ABC, abstractmethod

import numpy as np


class Space(ABC):
    @abstractmethod
    def random(self):
        raise NotImplementedError('Subclasses must implement random()')


class DiscreteSpace:
    def __init__(self, n: int):
        self.n = n

    def random(self):
        return np.random.randint(self.n)


class ContinuousSpace:
    def __init__(self, dim: int, low: np.ndarray, high: np.ndarray):
        self.dim = dim
        self.low = low
        self.high = high

    def random(self):
        return np.random.normal((self.high + self.low) / 2)

