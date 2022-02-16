from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from environments.environment import Environment
from learner.utils.decaying_variable import DecayingVariable


class Actor:
    def __init__(self,
                 environment: Environment,
                 discount: float = 0.7,
                 start_learning_rate: float = 1.0,
                 end_learning_rate: float = 0.1,
                 learning_rate_decay: float = 0.05,
                 start_epsilon: float = 1.0,
                 end_epsilon: float = 0.1,
                 epsilon_decay: float = 0.05,
                 trace_decay: float = 0.6):
        """
        :param environment: Environment object which the actor can interact with.
        :param discount: Discount
        :param start_learning_rate:
        :param end_learning_rate:
        :param learning_rate_decay:
        :param start_epsilon:
        :param end_epsilon:
        :param epsilon_decay:
        :param trace_decay:
        """

        self.environment: Environment = environment
        self.discount: float = discount
        self.learning_rate: DecayingVariable = DecayingVariable(start_learning_rate,
                                                                end_learning_rate,
                                                                learning_rate_decay)
        self.epsilon: DecayingVariable = DecayingVariable(start_epsilon,
                                                          end_epsilon,
                                                          epsilon_decay)
        self.trace_decay = trace_decay

        self.pi: np.ndarray = self._initialize_pi()
        self.eligibility: np.ndarray = np.zeros(self.pi.shape)
        self.episode_mask: np.ndarray = np.zeros(self.pi.shape, dtype=bool)

    def _initialize_pi(self) -> np.ndarray:
        """Initializes policy table.

        This method checks which actions are illegal in the environment and sets those entries to np.nan.

        :return: Policy table with zeros for legal SAPs and np.nan for illegal SAPs
        """

        pi = np.zeros(self.environment.state_shape + (self.environment.actions,))
        for state in np.ndindex(pi.shape[:-1]):
            for action, _ in enumerate(pi[state]):
                if not self.environment.action_legal_in_state(action, state):
                    pi[state][action] = np.nan
        return pi

    def choose_action(self, state: tuple, episode: Optional[int] = None) -> int:
        """Chooses action using an epsilon greedy scheme.

        This method chooses either:
        1) A random legal action (exploration)
        2) The "optimal" action in the current state (exploitation)

        If episode is not provided, epsilon will be set to zero,
        meaning the optimal action (according to the policy) will be taken.

        :param state: Current state.
        :param episode: Episode number. Used to decay epsilon.
        :return: Action chosen by the epsilon greedy algorithm.
        """

        if episode is not None and np.random.random() < self.epsilon(episode):
            non_nan_actions = np.argwhere(~np.isnan(self.pi[state])).flatten()
            action = np.random.choice(non_nan_actions)
        else:
            action = np.nanargmax(self.pi[state])

        if episode is not None:
            self.episode_mask[state][action] = True
            self.eligibility[state][action] = 1

        return action

    def update_pi(self, delta: float, episode: int) -> None:
        """Updates policy table PI and eligibility traces.

        :param delta: Temporal difference error (TD error).
        :param episode: Episode number. Used to decay learning rate.
        """

        self.pi[self.episode_mask] += self.learning_rate(episode) * delta * self.eligibility[self.episode_mask]
        self.eligibility[self.episode_mask] *= self.trace_decay * self.discount

    def reset(self) -> None:
        """Reset eligibilities and episode mask (which states have been visited)."""

        self.eligibility = np.zeros(self.pi.shape)
        self.episode_mask = np.zeros(self.pi.shape, dtype=bool)

    def visualize_strategy(self) -> None:
        """Visualizes strategy if policy table is two-dimensional."""

        if len(self.pi.shape) != 2:
            raise ValueError('Policy table (PI) must be two dimensional to visualize.')

        mask = np.all(np.isnan(self.pi), axis=1)  # Mask out states with no valid actions (all NaN)
        plt.plot(np.nanargmax(self.pi[~mask], axis=1))
        plt.show()

