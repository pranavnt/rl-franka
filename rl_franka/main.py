import time

import mujoco
import mujoco.viewer

import torch
import torch.nn as nn
import torch.nn.functional as F

model = mujoco.MjModel.from_xml_path("./mujoco_mengaerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

action_scale = 0.01

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)  # Use softmax instead of tanh
        return x

def calculate_reward(data):
    hand_z_position = data.xpos[-1, 2]
    reward = -abs(hand_z_position)
    return reward

def compute_returns(rewards):
    discounted_returns = []
    running_return = 0
    for reward in reversed(rewards):
        running_return = reward + running_return * 0.99
        discounted_returns.append(running_return)
    return discounted_returns[::-1]

policy_network = PolicyNetwork(input_size=7, output_size=7)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)

num_episodes = 10  # Define the number of episodes

for episode in range(num_episodes):
    # Reset environment and policy gradients at the start of each episode
    mujoco.mj_resetData(model, data)
    state = torch.tensor(data.qpos[:7], dtype=torch.float32)
    optimizer.zero_grad()

    # Initialize variables for rewards and log probabilities
    rewards = []
    log_probs = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running and time.time() - start < 30: 
            state = torch.tensor(data.qpos[:7], dtype=torch.float32)
            
            action_probs = policy_network(state)

            action = torch.distributions.Categorical(action_probs).sample()

            log_prob = torch.distributions.Categorical(action_probs).log_prob(action)


            data.ctrl[:7] += action.detach().numpy() * action_scale
            mujoco.mj_step(model, data)
            
            reward = calculate_reward(data)
            print(reward)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = torch.tensor(data.qpos[:7], dtype=torch.float32)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
            viewer.sync()

    returns = compute_returns(rewards)  
    loss = -torch.stack(log_probs) * torch.tensor(returns)
    loss = loss.sum()
    loss.backward()
    optimizer.step()

