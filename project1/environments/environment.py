from abc import ABC, abstractmethod


class Environment(ABC):
    """Abstract environment class used as a common interface for different environments/simworlds.

    Interface for environments:
    * State is a tuple.
        The interface should be a tuple even if the internals of the environment deals with one dimension.
    * Actions is an integer.
        This should account for all possible actions, and doesn't care if actions are illegal, but the environment
        must implement functionality to check which actions are legal in given states. Applications using the
        environment could check which actions are legal, but the environments can also punish if illegal actions
        are taken. Illegal actions should not move the state of the environment.
    """

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

    @abstractmethod
    def action_legal_in_state(self, action: int, state: tuple):
        """Checks whether an action is legal in a given state.

        :param action: Action to check.
        :param state: State to check.
        :return: Whether the action is legal in the given state.
        """

        raise NotImplementedError('Subclasses must implement action_legal_in_state()')

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        """The shape of the state space.

        :return: A tuple describing the shape of the state space.
        """

        raise NotImplementedError('Subclasses must implement state_shape property')

    @property
    @abstractmethod
    def actions(self) -> int:
        """The actions that can be performed.

        :return: Number of total actions.
        """

        raise NotImplementedError('Subclasses must implement actions property')

    @abstractmethod
    def visualize(self) -> None:
        """Visualizes the state history."""

        raise NotImplementedError('Subclasses must implement visualize() class method')


