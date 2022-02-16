import numpy as np

from learner.critics.critic import Critic


class TableCritic(Critic):
    def __init__(self, trace_decay: float = 0.6, *args, **kwargs):
        """
        :param trace_decay: Decay rate for eligibility traces.
        """

        super().__init__(*args, **kwargs)
        self.trace_decay = trace_decay
        self.v: np.ndarray = np.zeros(self.environment.state_shape)
        self.eligibility: np.ndarray = np.zeros(self.v.shape)
        self.episode_mask: np.ndarray = np.zeros(self.v.shape, dtype=bool)

    def get_delta(self, state: tuple, reward: float, next_state: tuple) -> float:
        """Computes the temporal difference error (delta/TD_error) based on state, reward, and next_state

        The delta is a measure of how good of an estimate we have of V(S).
        Small delta -> V(S) is currently pretty good estimate (small surprise)
        Large delta -> V(S) is not that good of an estimate (large surprise)

        :param state: Current state
        :param reward: Reward at next state
        :param next_state: Next state
        :return: Temporal difference error
        """

        delta: float = reward + self.discount * self.v[next_state] - self.v[state]
        self.episode_mask[state] = True
        self.eligibility[state] = 1
        return delta

    def update_v(self, delta: float, episode: int) -> None:
        """Updates value function V using the temporal difference error delta.

        :param delta: Temporal difference error
        :param episode: Episode nubmer. Used to decay learning rate.
        """

        self.v[self.episode_mask] += self.learning_rate(episode) * delta * self.eligibility[self.episode_mask]
        self.eligibility[self.episode_mask] *= self.discount * self.trace_decay

    def reset(self) -> None:
        """Resets eligibility."""

        self.eligibility = np.zeros(self.v.shape)
        self.episode_mask = np.zeros(self.v.shape, dtype=bool)


if __name__ == '__main__':
    t: TableCritic = TableCritic()
