import math
import random
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from environments.environment import Environment


class CartPole(Environment):
    def __init__(self,
                 L: float = 0.5,
                 m_p: float = 0.1,
                 m_c: float = 1.0,
                 g: float = -9.8,
                 F: float = 10.0,
                 theta_max: float = 0.21,
                 x_min: float = -2.4,
                 x_max: float = 2.4,
                 timestep_delta: float = 0.02,
                 buckets: Optional[tuple] = None,
                 *args,
                 **kwargs
                 ):
        """
        :param L: Length of pole.
        :param m_p: Mass of pole.
        :param m_c: Mass of cart.
        :param g: Gravitational constant.
        :param F: Bang-force magnitude.
        :param theta_max: Max radians the pole can be angled (absolute).
        :param x_min: Minimum horizontal position of cart.
        :param x_max: Maximum horizontal position of cart.
        :param timestep_delta: Step size between timesteps.
        :param buckets: If specified, the state is returned in bucketized form.
        """

        super().__init__(*args, **kwargs)
        self.L: float = L
        self.m_p: float = m_p
        self.m_c: float = m_c
        self.g: float = g
        self.F: float = F
        self.theta_max: float = theta_max
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.timestep_delta: float = timestep_delta
        self.buckets = buckets

        self.current_timestep: int = 0

        self.state = None
        self.low = np.array([x_min * 2, -1, -theta_max * 2, -math.radians(50)])
        self.high = np.array([x_max * 2, 1, theta_max * 2, math.radians(50)])

        self.state_history = []

    def bucketize_state(self, state: np.ndarray) -> tuple:
        """Transforms a continuous state into a discrete form and places the state in buckets.

        :param state: A continuous state.
        :return: A discrete/bucketized state.
        """

        bucketized = []
        for i in range(len(state)):
            low = self.low[i]
            high = self.high[i]
            scale = (state[i] + abs(low)) / (high - low)
            bucketized_value = int(round((self.buckets[i] - 1) * scale))
            bucketized_value = min(self.buckets[i] - 1, max(0, bucketized_value))
            bucketized.append(bucketized_value)
        return tuple(bucketized)

    def initialize(self) -> list:
        """Initializes environment/state and returns the initialized state.

        :return: The initial state.
        """

        self.current_timestep = 0
        theta = random.uniform(-self.theta_max, self.theta_max)
        x = (self.x_max + self.x_min) / 2
        self.state = [x, 0.0, theta, 0.0]
        self.state_history = []
        if self.store_states:
            self.state_history.append(self.state)
        return self.bucketize_state(self.state) if self.buckets else self.state

    def next(self, action: int) -> tuple[tuple, float, bool]:
        """Applies action to the environment, moving it to the next state.

        :param action: The action to perform
        :return: (next_state, reward, finished)
                    next_state: the current state of the environment after applying the action
                    reward: a numerical reward for moving to the state
                    finished: boolean specifying if the environment has reached some terminal condition
        """

        self.current_timestep += 1

        x, d_x, theta, d_theta = self.state
        B = self.F if action == 1 else -self.F

        # dd_theta
        term1 = math.cos(theta) * ((-B - self.m_p * self.L * d_theta * math.sin(theta)) / (self.m_p + self.m_c))
        term2 = (math.cos(theta) ** 2) * (self.m_p / (self.m_p + self.m_c))
        dd_theta = (self.g * math.sin(theta) + term1) / (self.L * ((4 / 3) - term2))

        # dd_x
        term3 = (d_theta ** 2) * math.sin(theta) - dd_theta * math.cos(theta)
        dd_x = (B + self.m_p * self.L * term3) / (self.m_p + self.m_c)

        # Update
        x += self.timestep_delta * d_x
        d_x += self.timestep_delta * dd_x
        theta += self.timestep_delta * d_theta
        d_theta += self.timestep_delta * dd_theta
        self.state = [x, d_x, theta, d_theta]
        if self.store_states:
            self.state_history.append(self.state)
        return self.bucketize_state(self.state) if self.buckets else self.state, 1.0, self.finished

    @property
    def finished(self) -> bool:
        """Checks whether the environment is finished/terminated.

        :return: Whether or not the environment is finished/terminated.
        """

        x, _, theta, _ = self.state
        return x >= self.x_max or x <= self.x_min or \
            abs(theta) >= self.theta_max or \
            self.current_timestep >= self.n_timesteps

    @property
    def state_shape(self) -> tuple:
        """The shape of the state space

        :return: A tuple describing the shape of the state space.
        """

        return (np.inf,) * 4 if self.buckets is None else self.buckets

    @property
    def actions(self) -> int:
        """The actions that can be performed.

        :return: Number of total actions
        """

        return 2

    def action_legal_in_state(self, action: int, state: tuple):
        """Checks whether an action is legal in a given state.

        :param action: Action to check.
        :param state: State to check.
        :return: Whether the action is legal in the given state.
        """

        return action in [0, 1]

    def visualize(self) -> None:
        """Visualizes the state history."""

        thetas = np.array(self.state_history)[:, 2]
        plt.plot(thetas)
        plt.show()





if __name__ == '__main__':
    c = CartPole()
    c.initialize()
    print(c.next(1))
    print(c.next(2))
