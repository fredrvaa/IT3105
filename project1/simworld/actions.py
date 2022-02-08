import numpy as np


class Actions:
    def __init__(self, n: int):
        self.n = n

    def random(self):
        return np.random.randint(self.n)