from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from simworld.cartpole import CartPole
from simworld.towers_of_hanoi import TowersOfHanoi


class Agent:
    def __init__(self,
                 env,
                 n_episodes: int = 2000,
                 discount: float = 1.0,
                 min_learning_rate: float = 0.1,
                 min_epsilon: float = 0.1,
                 decay: int = 25,
                 ):
        self.env = env
        self.n_episodes = n_episodes
        self.discount = discount
        self.min_learning_rate = min_learning_rate
        self.min_epsilon = min_epsilon
        self.decay = decay

        self.env = env
        self.Q = np.zeros(self.env.state_shape + (env.actions.n,))

        self.e = 0

        self.steps = np.zeros(n_episodes, dtype=int)
        self.best_steps = None

    @property
    def epsilon(self):
        return max(self.min_epsilon, min(1., 1. - np.log10((self.e + 1) / self.decay)))

    @property
    def learning_rate(self):
        return max(self.min_learning_rate, min(1., 1. - np.log10((self.e + 1) / self.decay)))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.actions.random()
        else:
            return np.argmax(self.Q[state])

    def update_q(self, state, action, reward, next_state):
        TD_error = reward + self.discount * np.max(self.Q[next_state]) - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * TD_error

    def fit(self):
        for e in range(self.n_episodes):
            self.e = e
            current_state = self.env.initialize()

            finished = False
            while not finished:
                self.steps[e] += 1
                action = self.choose_action(current_state)
                next_state, reward, finished = self.env.next(action)
                self.update_q(current_state, action, reward, next_state)
                current_state = next_state
            print(f'Finished episode {e} in {self.steps[e]} steps')
            state_history = self.env.state_history
            if self.best_steps is None or len(state_history) < len(self.best_steps):
                self.best_steps = state_history

    def visualize(self):
        plt.plot(self.steps)
        plt.show()


if __name__ == '__main__':
    #env = CartPole(buckets=(3, 3, 6, 6))
    env = TowersOfHanoi(n_disks=4, n_pegs=3)
    agent = Agent(env, discount=0.9)
    agent.fit()
    agent.visualize()
    print(agent.best_steps)
    env.visualize(agent.best_steps)