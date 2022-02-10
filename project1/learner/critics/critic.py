from abc import ABC, abstractmethod

from environments.environment import Environment
from learner.utils.decaying_variable import DecayingVariable


class Critic(ABC):
    def __init__(self,
                 environment: Environment,
                 discount: float = 0.7,
                 start_learning_rate: float = 1.0,
                 end_learning_rate: float = 0.1,
                 learning_rate_decay: float = 0.05):
        self.environment: Environment = environment
        self.discount: float = discount
        self.learning_rate: DecayingVariable = DecayingVariable(start_learning_rate,
                                                                end_learning_rate,
                                                                learning_rate_decay)

    @abstractmethod
    def get_delta(self, state: tuple, reward: float, next_state: tuple) -> float:
        raise NotImplementedError('Subclasses must implement get_delta()')

    @abstractmethod
    def update_v(self, delta: float, episode: int) -> None:
        raise NotImplementedError('Subclasses must implement update_v()')

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError('Subclasses must implement reset()')
