from abc import ABC, abstractmethod

from environments.actions import Actions


class Environment(ABC):
    def __init__(self):
        self.store_states = False

    @abstractmethod
    def initialize(self) -> tuple:
        raise NotImplementedError('Subclasses must implement initialize()')

    @abstractmethod
    def next(self, action: int) -> tuple[tuple, float, bool]:
        raise NotImplementedError('Subclasses must implement next()')

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        raise NotImplementedError('Subclasses must implement state_shape property')

    @property
    @abstractmethod
    def actions(self) -> Actions:
        raise NotImplementedError('Subclasses must implement actions property')

    @abstractmethod
    def visualize(self) -> None:
        raise NotImplementedError('Subclasses must implement visualize() class method')


