import numpy as np
import torch

from learner.critics.critic import Critic
from learner.utils.network import Network


class NetworkCritic(Critic):
    def __init__(self, layer_sizes: list[int] = [6, 4, 4], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_lenghts = tuple([len(format(s, 'b')) for s in self.environment.state_shape])
        self.nn_input_size = sum(self.binary_lenghts)
        self.v: Network = Network(self.nn_input_size, layer_sizes)
        print(self.v)
        self.optimizer = torch.optim.Adam(self.v.parameters(), lr=self.learning_rate())
        self.batch_size = 5
        self.batch_losses = []

    def encode_state(self, state: tuple) -> np.ndarray:
        encoded_state = ''
        for i, s in enumerate(state):
            binary_state = format(s, f'0{self.binary_lenghts[i]}b')
            encoded_state += binary_state
        return np.array(list(encoded_state), dtype=int)

    def get_delta(self, state: tuple, reward: float, next_state: tuple) -> float:
        with torch.no_grad():
            next_state = torch.FloatTensor(self.encode_state(next_state))
            y = reward + self.discount * self.v(next_state)

        state = torch.FloatTensor(self.encode_state(state))
        y_hat = self.v(state)

        delta = y - y_hat
        return delta

    def update_v(self, delta, episode: int) -> None:
        self.batch_losses.append(delta)
        if len(self.batch_losses) >= self.batch_size:
            self.optimizer.param_groups[0]['lr'] = self.learning_rate(episode)
            loss = torch.cat(self.batch_losses).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            #print(loss)
            self.optimizer.step()
            self.batch_losses = []

    def reset(self) -> None:
        pass

