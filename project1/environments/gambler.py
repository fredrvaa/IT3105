import numpy as np
from matplotlib import pyplot as plt

from environments.environment import Environment


class Gambler(Environment):
    def __init__(self, win_probability: float = 0.6, goal_money: int = 100, *args, **kwargs):
        """
        :param win_probability: Probability that the gambler wins at each timestep.
        :param goal_money: The money the gambler must accumulate before he wins.
        """

        super().__init__(*args, **kwargs)
        self.win_probability = win_probability
        self.goal_money: int = goal_money
        self.current_timestep: int = 0
        self.state = None
        self.state_history = []

    def initialize(self):
        """Initializes environment/state and returns the initialized state.

        :return: The initial state.
        """

        self.state = np.random.randint(1, self.goal_money)
        self.state_history = []
        if self.store_states:
            self.state_history.append(self.state)

        return self.state,

    def action_legal_in_state(self, action: int, state: tuple):
        """Checks whether an action is legal in a given state.

        :param action: Action to check.
        :param state: State to check.
        :return: Whether the action is legal in the given state.
        """
        bet = action + 1

        return bet <= state[0] and bet + state[0] <= self.goal_money

    def _is_won(self) -> bool:
        """Checks whether the gambler has won in the current state.

        :return: Whether or not the gambler has won.
        """

        return self.state == self.goal_money

    def _is_finished(self) -> bool:
        """Checks whether the environment has finished (terminated).

        :return: Whether or not environment is finished.
        """

        return self._is_won() or self.state == 0 or self.current_timestep >= self.n_timesteps

    def _perform_bet(self, bet: int) -> None:
        """Performs bet and updates state based on if the gambler won or lost.

        :param bet: The money that was bet.
        """

        if np.random.random() < self.win_probability:
            self.state += bet
        else:
            self.state -= bet

    def next(self, action: int):
        """Applies action to the environment, moving it to the next state.

        :param action: The action to perform
        :return: (next_state, reward, finished)
                    next_state: the current state of the environment after applying the action
                    reward: a numerical reward for moving to the state
                    finished: boolean specifying if the environment has reached some terminal condition
        """

        if self.action_legal_in_state(action, (self.state, )):
            bet = action + 1
            self._perform_bet(bet)
            if self._is_won():
                reward = 100
            else:
                reward = 0
            finished = self._is_finished()
        else:
            reward = -10
            finished = False
        if self.store_states:
            self.state_history.append(self.state)
        return (self.state, ), reward, finished

    @property
    def state_shape(self) -> tuple:
        """The shape of the state space

        :return: A tuple describing the shape of the state space.
        """

        return self.goal_money + 1,

    @property
    def actions(self) -> int:
        """The actions that can be performed.

        :return: Number of total actions
        """

        return int(self.goal_money / 2)

    def visualize(self) -> None:
        """Visualizes the state history."""

        plt.plot(np.array(self.state_history))
        plt.show()

