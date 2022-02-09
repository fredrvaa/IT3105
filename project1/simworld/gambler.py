import numpy as np

from simworld.actions import Actions
from simworld.simworld import Simworld


class Gambler(Simworld):
    def __init__(self, win_probability: float = 0.6, goal_money: int = 100, n_timesteps: int = 2000):
        self.win_probability = win_probability
        self.goal_money: int = goal_money
        self.n_timesteps: int = n_timesteps
        self.current_timestep: int = 0
        self.state = None
        self.state_shape = (self.goal_money + 1, )
        self.state_history = []
        self.actions: Actions = Actions(int(self.goal_money / 2))

    def initialize(self):
        self.state = np.random.randint(1, self.goal_money)
        self.state_history = [self.state]
        return self.state

    def _is_legal(self, bet: int):
        return bet <= self.state and bet + self.state <= self.goal_money

    def _is_won(self):
        return self.state == self.goal_money

    def _is_finished(self):
        return self._is_won() or self.state == 0 or self.current_timestep >= self.n_timesteps

    def _perform_bet(self, bet: int):
        if np.random.random() < self.win_probability:
            self.state += bet
        else:
            self.state -= bet

    def next(self, action: int):
        bet = action + 1
        if self._is_legal(bet):
            self._perform_bet(bet)
            if self._is_won():
                reward = 100
            else:
                reward = 0
            finished = self._is_finished()
        else:
            reward = -1
            finished = False
        self.state_history.append(self.state)
        return self.state, reward, finished
