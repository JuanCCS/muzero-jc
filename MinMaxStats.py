import numpy as np

class MinMaxStats:
    def __init__():
        self.maximum = -np.inf
        self.minimum = np.inf

    def update(self, value:float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

