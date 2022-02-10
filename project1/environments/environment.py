from abc import ABC, abstractmethod

from environments.actions import Actions


class Environment(ABC):
    @abstractmethod
    def initialize(self) -> tuple:
        raise NotImplementedError('Subclasses must implement initialize()')

    @abstractmethod
    def next(self) -> tuple:
        raise NotImplementedError('Subclasses must implement next()')

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        return

    @property
    @abstractmethod
    def actions(self) -> Actions:
        return


