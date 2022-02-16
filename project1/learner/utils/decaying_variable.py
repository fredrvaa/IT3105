from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DecayingVariable:
    """Utility class for decaying a variable based on episode."""

    start_value: float
    end_value: Optional[float] = None
    decay: Optional[float] = None

    def __call__(self, episode: Optional[int] = None):
        """Returns the decayed value of the variable based on the episode.

        :param episode: The episode used to calculate the variable value.
        :return: The decayed value.
        """

        if episode is None or self.end_value is None or self.decay is None:
            return self.start_value
        decayed = min(self.start_value, self.start_value - np.log10((episode + 1) * self.decay))
        return max(decayed, self.end_value)
