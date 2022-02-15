import numpy as np


class Actions:
    """Class used as a common interface for actions."""

    def __init__(self, n: int):
        """
        :param n: Number of possible actions
        """
        self.n = n

    def random(self) -> int:
        """Returns a random action.

        :return: Integer corresponding to an action.
        """

        return np.random.randint(self.n)
