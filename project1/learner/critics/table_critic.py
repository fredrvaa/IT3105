import numpy as np

from learner.critics.critic import Critic


class TableCritic(Critic):
    def __init__(self, trace_decay: float = 0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_decay = trace_decay
        self.v: np.ndarray = np.zeros(self.environment.state_shape)
        self.eligibility: np.ndarray = np.zeros(self.v.shape)
        self.episode_mask: np.ndarray = np.zeros(self.v.shape, dtype=bool)

    def get_delta(self, state: tuple, reward: float, next_state: tuple) -> float:
        delta: float = reward + self.discount * self.v[next_state] - self.v[state]
        self.episode_mask[state] = True
        self.eligibility[state] = 1
        return delta

    def update_v(self, delta: float, episode: int) -> None:
        self.v[self.episode_mask] += self.learning_rate(episode) * delta * self.eligibility[self.episode_mask]
        self.eligibility[self.episode_mask] *= self.discount * self.trace_decay

    def reset(self) -> None:
        self.eligibility = np.zeros(self.v.shape)
        self.episode_mask = np.zeros(self.v.shape, dtype=bool)


if __name__ == '__main__':
    t: TableCritic = TableCritic()
