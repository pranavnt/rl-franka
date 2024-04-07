import torch
import torch.optim as optim
from environments import MujocoEnv
from algorithms import REINFORCE
from policies import MLPPolicy
from utils import LinearScheduler

def calculate_reward(data):
    hand_pos = torch.tensor(data.body("hand").xpos)
    block_pos = torch.tensor(data.body("block").xpos)
    return -1 * torch.sum((hand_pos - block_pos))

env = MujocoEnv(model_path="./mujoco_mengaerie/franka_emika_panda/scene.xml", reward_func=calculate_reward, render=True)

policy = MLPPolicy(input_size=13, hidden_sizes=[64, 64], output_size=14)

optimizer = optim.Adam(policy.parameters(), lr=0.001)

lr_scheduler = LinearScheduler(start_value=0.001, end_value=0.0001, num_steps=1000)

algo = REINFORCE(optimizer=optimizer)

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

        env.sync()

        if step % 5000 == 0:
            print(str(step) + " " + str(reward))

    returns = REINFORCE._compute_returns(episode_rewards, 0.99)
    loss = -1 * REINFORCE._compute_loss(episode_logprobs, returns).sum()
    loss.backward()
    optimizer.step()
    
    lr = lr_scheduler.step()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

env.close()




# import time
# import mujoco
# import mujoco.viewer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# data = mujoco.MjData(model)

# action_scale = 0.01

# class PolicyNetwork(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, output_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.softmax(self.fc2(x), dim=-1)
#         return x

# def compute_returns(rewards):
#     discounted_returns = []
#     running_return = 0
#     for reward in reversed(rewards):
#         running_return = reward + running_return * 0.99
#         discounted_returns.append(running_return)
#     return discounted_returns[::-1]



# policy = PolicyNetwork(input_size=13, output_size=14)
# optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

# num_episodes = 10
# num_steps_per_episode = 30000

# for episode in range(num_episodes):
#     mujoco.mj_resetData(model, data)
#     state = torch.tensor(data.qpos[:7], dtype=torch.float32)
#     optimizer.zero_grad()

#     episode_rewards = []
#     episode_logprobs = []

#     curr_step = 0

#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         start = time.time()
#         while viewer.is_running and curr_step < num_steps_per_episode: 
#             state = torch.tensor(data.qpos[:7], dtype=torch.float32)
#             hand_pos = torch.tensor(data.body("hand").xpos, dtype=torch.float32)
#             block_pos = torch.tensor(data.body("block").xpos, dtype=torch.float32)

#             input_state = torch.cat([hand_pos, block_pos, state])
            
#             action_probs = policy(input_state)

#             action = torch.distributions.Categorical(action_probs).sample()

#             log_prob = torch.distributions.Categorical(action_probs).log_prob(action)

#             if action // 7 == 0:
#                 data.ctrl[action % 7] += action_scale
#             else:
#                 data.ctrl[action % 7] -= action_scale

#             mujoco.mj_step(model, data)
            
#             reward = calculate_reward(data)
#             episode_rewards.append(reward)
#             episode_logprobs.append(log_prob)
#             curr_step += 1

#             with viewer.lock():
#                 viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
#             viewer.sync()

#             if curr_step % 5000 == 0:
#                 print(str(curr_step) + " " + str(reward))
#                 print("hand pos: " + str(hand_pos) + " block pos: " + str(block_pos))

#     returns = compute_returns(episode_rewards)

#     optimizer.zero_grad()
#     loss = -torch.stack(episode_logprobs) * torch.tensor(returns)
#     loss = loss.sum()
#     loss.backward()
#     optimizer.step()