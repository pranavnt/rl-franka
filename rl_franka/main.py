import time

import mujoco
import mujoco.viewer

import torch
import torch.nn as nn
import torch.nn.functional as F

model = mujoco.MjModel.from_xml_path("./mujoco_mengaerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) 
        return x

def calculate_reward(data):
    hand_z_position = data.xpos[-1, 2]
    reward = -abs(hand_z_position)
    return reward

policy_network = PolicyNetwork(input_size=7, output_size=7)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30: 
        step_start = time.time()

        state = torch.tensor(data.qpos[:7], dtype=torch.float32)
        
        action = policy_network(state).detach().numpy()
        
        # Apply action
        data.ctrl[:7] = action

        mujoco.mj_step(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

