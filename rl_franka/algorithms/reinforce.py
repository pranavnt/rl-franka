import torch
from .base_algorithm import BaseAlgorithm

class REINFORCE(BaseAlgorithm):
    def calculate_loss(self, logprobs, rewards):
        returns = self.calculate_returns(rewards, gamma=0.99)
        loss = -(logprobs * returns).mean()
        return loss