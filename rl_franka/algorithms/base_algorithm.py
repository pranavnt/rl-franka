import abc
import torch

class BaseAlgorithm(abc.ABC):
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def calculate_loss(self, *args, **kwargs):
        ...

    @staticmethod
    def calculate_returns(rewards, gamma):
        returns = torch.zeros_like(rewards)
        return_so_far = 0
        for t in reversed(range(len(rewards))):
            return_so_far = rewards[t] + gamma * return_so_far
            returns[t] = return_so_far
        return returns

    @staticmethod
    def compute_advantages(rewards, values, gamma, gae_lambda):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + gamma * values[t+1] - values[t]
            gae = td_error + gamma * gae_lambda * gae
            advantages[t] = gae
        return advantages

    @staticmethod
    def normalize(data):
        return (data - data.mean()) / (data.std() + 1e-8)