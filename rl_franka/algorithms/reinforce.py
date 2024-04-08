import torch
from .base_algorithm import BaseAlgorithm

class REINFORCE(BaseAlgorithm):
    def calculate_loss(logprobs, rewards, gamma=0.999):
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards)
        returns = BaseAlgorithm.calculate_returns(rewards, gamma=gamma)
        return -torch.mean(torch.stack(logprobs) * returns).sum()

