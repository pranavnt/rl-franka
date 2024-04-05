import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_policy import BasePolicy

class MLPPolicy(BasePolicy):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPPolicy, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.model(obs)
        return F.softmax(logits, dim=-1)

    def act(self, obs):
        probs = self.forward(obs)
        action = torch.distributions.Categorical(probs).sample()
        return action.item()