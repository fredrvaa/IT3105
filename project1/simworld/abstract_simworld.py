from abc import ABC, abstractmethod
from enum import Enum


class Simworld(ABC):
    @abstractmethod
    def initialize(self):
        raise NotImplementedError('Subclasses must implement initialize()')

    @abstractmethod
    def next(self):
        raise NotImplementedError('Subclasses must implement next()')
