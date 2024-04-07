from abc import ABC, abstractmethod

class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass