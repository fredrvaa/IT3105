from abc import ABC, abstractmethod

from environments.environment import Environment
from learner.utils.decaying_variable import DecayingVariable


class Critic(ABC):
    """Abstract critic class which should be inherited from."""

    def __init__(self,
                 environment: Environment,
                 discount: float = 0.7,
                 start_learning_rate: float = 1.0,
                 end_learning_rate: float = 0.1,
                 learning_rate_decay: float = 0.05):
        """
        :param environment: Environment object that the critic observes.
        :param discount: Discount parameter used to train V(S).
        :param start_learning_rate: Learning rate at the start of training.
        :param end_learning_rate: Learning rate at the end of training.
        :param learning_rate_decay: Learning rate decay factor.
        """

        self.environment: Environment = environment
        self.discount: float = discount
        self.learning_rate: DecayingVariable = DecayingVariable(start_learning_rate,
                                                                end_learning_rate,
                                                                learning_rate_decay)

    @abstractmethod
    def get_delta(self, state: tuple, reward: float, next_state: tuple) -> float:
        """Computes the temporal difference error (delta/TD_error) based on state, reward, and next_state

        The delta is a measure of how good of an estimate we have of V(S).
        Small delta -> V(S) is currently pretty good estimate (small surprise)
        Large delta -> V(S) is not that good of an estimate (large surprise)

        :param state: Current state
        :param reward: Reward at next state
        :param next_state: Next state
        :return: Temporal difference error
        """

        raise NotImplementedError('Subclasses must implement get_delta()')

    @abstractmethod
    def update_v(self, delta: float, episode: int) -> None:
        """Updates value function V using the temporal difference error delta.

        :param delta: Temporal difference error
        :param episode: Episode nubmer. Used to decay learning rate.
        """

        raise NotImplementedError('Subclasses must implement update_v()')

    @abstractmethod
    def reset(self) -> None:
        """Resets variables that have to be reset before a new episode."""

        raise NotImplementedError('Subclasses must implement reset()')
