import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from simworld.simworld import Simworld
from simworld.actions import Actions
from simworld.spaces import ContinuousSpace, DiscreteSpace

@dataclass
class State:
    x: float
    d_x: float
    dd_x: float
    theta: float
    d_theta: float
    dd_theta: float


class CartPole(Simworld):
    def __init__(self,
                 L: float = 0.5,
                 m_p: float = 0.1,
                 m_c: float = 1.0,
                 g: float = 9.8,
                 F: float = 10.0,
                 theta_max: float = 0.21,
                 x_min: float = -2.4,
                 x_max: float = 2.4,
                 timestep_delta: float = 0.02,
                 n_timesteps: int = 300,
                 buckets: Optional[tuple] = None
                 ):
        self.L: float = L
        self.m_p: float = m_p
        self.m_c: float = m_c
        self.g: float = g
        self.F: float = F
        self.theta_max: float = theta_max
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.timestep_delta: float = timestep_delta
        self.n_timesteps: int = n_timesteps
        self.buckets = buckets

        self.current_timestep: int = 0

        self.state = None
        self.low = np.array([x_min * 2, -1, -theta_max * 2, -math.radians(50)])
        self.high = np.array([x_max * 2, 1, theta_max * 2, math.radians(50)])
        self.state_shape = (np.inf,) * 4 if buckets is None else buckets
        self.actions = Actions(2)

    def initialize(self) -> list:
        self.current_timestep = 0
        theta = random.uniform(-self.theta_max, self.theta_max)
        x = (self.x_max + self.x_min) / 2
        self.state = [x, 0.0, 0.0, theta]
        return self.bucketize_state(self.state) if self.buckets else self.state

    def next(self, action):
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

        return self.bucketize_state(self.state) if self.buckets else self.state, 1, self.finished

    @property
    def finished(self) -> bool:
        x, _, theta, _ = self.state
        return x >= self.x_max or x <= self.x_min or \
            abs(theta) >= self.theta_max or \
            self.current_timestep >= self.n_timesteps

    def bucketize_state(self, state: np.ndarray):
        bucketized = []
        for i in range(len(state)):
            low = self.low[i]
            high = self.high[i]
            scale = (state[i] + abs(low)) / (high - low)
            bucketized_value = int(round((self.buckets[i] - 1) * scale))
            bucketized_value = min(self.buckets[i] - 1, max(0, bucketized_value))
            bucketized.append(bucketized_value)
        return tuple(bucketized)


if __name__ == '__main__':
    c = CartPole()
    c.initialize()
    print(c.next(1))
    print(c.next(2))
