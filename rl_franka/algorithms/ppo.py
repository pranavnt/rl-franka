import torch
from .base_algorithm import BaseAlgorithm

class PPO(BaseAlgorithm):
    def calculate_loss(self, *args, **kwargs):
        return super().calculate_loss(*args, **kwargs)