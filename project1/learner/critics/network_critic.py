import numpy as np
import torch

from learner.critics.critic import Critic
from learner.utils.network import Network


class NetworkCritic(Critic):
    """Critic using a pytorch neural network."""

    def __init__(self, layer_sizes: list[int] = [6, 4, 4], batch_size: int = 1, *args, **kwargs):
        """
        :param layer_sizes: Hidden layer sizes. Each entry in the list specifies the size of a hidden layer.
        :param batch_size: How many losses should be accumulated before stepping the optimizer.
        """

        super().__init__(*args, **kwargs)
        self.binary_lenghts = tuple([len(format(s, 'b')) for s in self.environment.state_shape])
        self.nn_input_size = sum(self.binary_lenghts)
        self.v: Network = Network(self.nn_input_size, layer_sizes)
        self.optimizer = torch.optim.Adam(self.v.parameters(), lr=self.learning_rate())
        self.batch_size = batch_size
        self.batch_losses = []

    def encode_state(self, state: tuple) -> np.ndarray:
        """Encodes a tupled state to a bit list.

        Example: (2,3,4) -> [(0, 1, 0), (0, 1, 1), (1, 0, 0)] (Parenthesis for visualizing each integer)

        :param state: State in the form of a tuple.
        :return: State in the form of a bit list.
        """

        encoded_state = ''
        for i, s in enumerate(state):
            binary_state = format(s, f'0{self.binary_lenghts[i]}b')
            encoded_state += binary_state
        return np.array(list(encoded_state), dtype=int)

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

        with torch.no_grad():
            next_state = torch.FloatTensor(self.encode_state(next_state))
            y = reward + self.discount * self.v(next_state)

        state = torch.FloatTensor(self.encode_state(state))
        y_hat = self.v(state)

        delta = y - y_hat
        return delta

    def update_v(self, delta, episode: int) -> None:
        """Updates value function V using the temporal difference error delta.

       :param delta: Temporal difference error
       :param episode: Episode nubmer. Used to decay learning rate.
       """

        self.batch_losses.append(delta)
        if len(self.batch_losses) >= self.batch_size:
            self.optimizer.param_groups[0]['lr'] = self.learning_rate(episode)
            loss = torch.cat(self.batch_losses).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.batch_losses = []

    def reset(self) -> None:
        """Nothing has to be reset for the NetworkCritic in between episodes."""
        pass

