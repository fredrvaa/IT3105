from abc import ABC, abstractmethod

from environments.actions import Actions


class Environment(ABC):
    """Abstract environment class used as a common interface for different environments/simworlds."""

    def __init__(self, n_timesteps: int = 2000):
        """
        :param n_timesteps: Max number of timesteps that can be performed before environment is terminated.
        """
        self.store_states: int = False
        self.n_timesteps: int = n_timesteps

    @abstractmethod
    def initialize(self) -> tuple:
        """Initializes environment/state and returns the initialized state.

        :return: The initial state.
        """

        raise NotImplementedError('Subclasses must implement initialize()')

    @abstractmethod
    def next(self, action: int) -> tuple[tuple, float, bool]:
        """Applies action to the environment, moving it to the next state.

        :param action: The action to perform
        :return: (next_state, reward, finished)
                    next_state: the current state of the environment after applying the action
                    reward: a numerical reward for moving to the state
                    finished: boolean specifying if the environment has reached some terminal condition
        """

        raise NotImplementedError('Subclasses must implement next()')

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        """The shape of the state space.

        :return: A tuple describing the shape of the state space.
        """

        raise NotImplementedError('Subclasses must implement state_shape property')

    @property
    @abstractmethod
    def actions(self) -> Actions:
        """The actions that can be performed.

        :return: A Actions object specifying the environment's actions.
        """

        raise NotImplementedError('Subclasses must implement actions property')

    @abstractmethod
    def visualize(self) -> None:
        """Visualizes the state history."""

        raise NotImplementedError('Subclasses must implement visualize() class method')


