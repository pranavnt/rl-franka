import torch
import torch.optim as optim
import glob
from torch.utils.data import TensorDataset, DataLoader
from environments import MujocoEnv
import numpy as np
from algorithms import REINFORCE
from policies import MLPPolicy
from utils import LinearScheduler

def distance_reward(data):
    hand_pos = torch.tensor(data.body("hand").xpos)
    block_pos = torch.tensor(data.body("block").xpos)
    return -1 * torch.sum((hand_pos - block_pos))

def reinforce_main():
    env = MujocoEnv(model_path="./mujoco_mengaerie/franka_emika_panda/scene.xml", reward_func=distance_reward, render=True)

    policy = MLPPolicy(input_size=13, hidden_sizes=[64, 64], output_size=14)

    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    lr_scheduler = LinearScheduler(start_value=0.001, end_value=0.0001, num_steps=1000)

    num_episodes = 1000
    num_steps = 30000

    for episode in range(num_episodes):
        episode_rewards = []
        episode_logprobs = []

        env.reset()

        for step in range(num_steps):
            obs = env.get_obs()

            action, probs, logprobs = policy.sample(obs)

            obs, reward, _, _ = env.step(action)

            episode_rewards.append(reward)
            episode_logprobs.append(logprobs)

            if (step + 1) % 1000 == 0 or step == num_steps - 1:  
                loss = REINFORCE.calculate_loss(episode_logprobs, episode_rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_rewards = []
                episode_logprobs = []

            lr = lr_scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    env.close()

def behavior_cloning_main(data_dir="./data"):
    data_files = glob.glob(data_dir + "/*.npy")

    data_len = 0

    for data_file in data_files:
        data = np.load(data_file, allow_pickle=True)
        data_len += len(data)

    inputs = torch.zeros(data_len, 13)
    targets = torch.zeros(data_len, 14)

    idx = 0
    for data_file in data_files:
        data = np.load(data_file, allow_pickle=True)
        for x, y in data:
            inputs[idx] = x
            targets[idx] = y
            idx += 1

    policy = MLPPolicy(input_size=13, hidden_sizes=[64, 64], output_size=14)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    dataset = TensorDataset(inputs, targets)
    batch_size = 64  # You can adjust the batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 1000

    for epoch in range(num_epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            pred = policy(x_batch)
            loss = (pred - y_batch).pow(2).sum()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} loss: {loss.item()}")

# def ppo_main():
#     env = MujocoEnv(model_path="./mujoco_mengaerie/franka_emika_panda/scene.xml", reward_func=distance_reward, render=True)

#     algo = PPO(optimizer=optimizer)

if __name__ == "__main__":
    reinforce_main()

