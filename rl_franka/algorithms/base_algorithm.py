from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def evaluate(self, env, policy, num_episodes):
        pass