from abc import ABC

class Environment(ABC):

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
