from abc import ABC, abstractmethod


class Simworld(ABC):
    @abstractmethod
    def initialize(self):
        raise NotImplementedError('Subclasses must implement initialize()')

    @abstractmethod
    def next(self):
        raise NotImplementedError('Subclasses must implement next()')
