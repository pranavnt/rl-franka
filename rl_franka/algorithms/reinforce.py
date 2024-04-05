import torch
from .base_algorithm import BaseAlgorithm

class REINFORCE(BaseAlgorithm):
    def __init__(self, optimizer, gamma=0.99):
        self.optimizer = optimizer
        self.gamma = gamma

    def train(self, env, policy, num_episodes):
        for episode in range(num_episodes):
            obs = env.reset()
            episode_rewards = []
            episode_logprobs = []
            done = False
            while not done:
                action_probs = policy(obs)
                action = torch.distributions.Categorical(action_probs).sample()
                log_prob = torch.distributions.Categorical(action_probs).log_prob(action)
                obs, reward, done, _ = env.step(action.item())
                episode_rewards.append(reward)
                episode_logprobs.append(log_prob)

            returns = self._compute_returns(episode_rewards)
            loss = self._compute_loss(episode_logprobs, returns)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, env, policy, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action_probs = policy(obs)
                action = torch.argmax(action_probs)
                obs, reward, done, _ = env.step(action.item())
                episode_reward += reward
            total_rewards.append(episode_reward)
        return total_rewards

    def _compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def _compute_loss(self, logprobs, returns):
        return -torch.mean(torch.stack(logprobs) * returns)