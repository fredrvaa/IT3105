import numpy as np
from torch import nn


class Network(nn.Module):
    """Wrapper around nn.Module to specify custom pytorch network."""

    def __init__(self, input_size: int, layer_sizes: list[int]):
        """
        :param input_size: Size of input into network.
        :param layer_sizes: List of all layer sizes in network.
        """

        super().__init__()
        layer_sizes.append(1)
        layer_stack = []
        layer_stack.append(nn.Linear(input_size, layer_sizes[0]))
        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layer_stack.append(nn.ReLU())
            lin = nn.Linear(in_size, out_size)
            nn.init.xavier_uniform(lin.weight)
            layer_stack.append(lin)

        self.layer_stack = nn.Sequential(*layer_stack)

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight)
        self.layer_stack.apply(weights_init)

    def forward(self, x: np.ndarray) -> float:
        """Forwards x through the network.

        :param x: Input to network.
        :return: Output after forward pass.
        """

        return self.layer_stack(x)
