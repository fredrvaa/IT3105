from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from environments.environment import Environment
from learner.actors.actor import Actor
from learner.critics.critic import Critic


class ActorCritic:
    def __init__(self,
                 environment: Environment,
                 actor: Actor,
                 critic: Critic):
        self.environment: Environment = environment
        self.actor: Actor = actor
        self.critic: Critic = critic

        self.steps: Optional[np.ndarray] = None

    def fit(self, n_episodes: int = 300):
        self.steps = np.zeros(n_episodes, dtype=int)
        for episode in range(n_episodes):
            self.actor.reset()
            self.critic.reset()

            state = self.environment.initialize()

            finished = False
            while not finished:
                self.steps[episode] += 1
                action = self.actor.choose_action(state, episode)

                next_state, reward, finished = self.environment.next(action)

                delta = self.critic.get_delta(state, reward, next_state)
                self.critic.update_v(delta, episode)
                self.actor.update_pi(delta, episode)

                state = next_state

            print(f'Finished episode {episode} after {self.steps[episode]} steps')

    def visualize_fit(self):
        plt.plot(self.steps)
        plt.show()

    def run(self, visualize: bool = True):
        steps: int = 0
        state = self.environment.initialize()
        self.environment.store_states = visualize
        finished = False
        while not finished:
            steps += 1
            action = self.actor.choose_action(state)
            state, _, finished = self.environment.next(action)

        print(f'Finished run after {steps} steps')
        if visualize:
            self.environment.visualize()
