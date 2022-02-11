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
        self.environment: Environment = environment
        self.discount: float = discount
        self.learning_rate: DecayingVariable = DecayingVariable(start_learning_rate,
                                                                end_learning_rate,
                                                                learning_rate_decay)
        self.epsilon: DecayingVariable = DecayingVariable(start_epsilon,
                                                          end_epsilon,
                                                          epsilon_decay)
        self.trace_decay = trace_decay

        self.pi: np.ndarray = np.zeros(environment.state_shape + (environment.actions.n,))
        self.eligibility: np.ndarray = np.zeros(self.pi.shape)
        self.episode_mask: np.ndarray = np.zeros(self.pi.shape, dtype=bool)

    def choose_action(self, state: tuple, episode: Optional[int] = None) -> int:
        if episode is not None and np.random.random() < self.epsilon(episode):
            action = self.environment.actions.random()
        else:
            action = np.argmax(self.pi[state])

        if episode is not None:
            self.episode_mask[state][action] = True
            self.eligibility[state][action] = 1

        return action

    def update_pi(self, delta: float, episode: int) -> None:
        self.pi[self.episode_mask] += self.learning_rate(episode) * delta * self.eligibility[self.episode_mask]
        self.eligibility[self.episode_mask] *= self.trace_decay * self.discount
        #print(self.pi.shape)

    def reset(self) -> None:
        self.eligibility = np.zeros(self.pi.shape)
        self.episode_mask = np.zeros(self.pi.shape, dtype=bool)

    def visualize_strategy(self) -> None:
        if len(self.pi.shape) != 2:
            raise ValueError('Policy table (PI) must be two dimensional to visualize.')

        print(self.pi.shape)
        plt.plot(np.argmax(self.pi, axis=1))
        plt.show()

