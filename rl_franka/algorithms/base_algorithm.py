from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def train(self, env, policy, num_episodes):
        pass

    @abstractmethod
    def evaluate(self, env, policy, num_episodes):
        pass