from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def initialize(self):
        raise NotImplementedError('Subclasses must implement initialize()')

    @abstractmethod
    def next(self):
        raise NotImplementedError('Subclasses must implement next()')

    @property
    @abstractmethod
    def state_shape(self):
        return

    @property
    @abstractmethod
    def actions(self):
        return


