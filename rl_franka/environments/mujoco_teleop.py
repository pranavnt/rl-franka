import mujoco
import mujoco.viewer
import torch
import numpy as np
import os
from base_env import BaseEnv

import numpy as np
import mujoco
import mujoco.viewer

class MujocoTeleopEnv(BaseEnv):
    def __init__(self, model_path, reward_func, render=False):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.reward_func = reward_func
        self.viewer = None
        self.collected_data = []
        self.window = None
        self.action = None
        self.paused = False
        self.action = None
        self.file_name = "collected_data.npy"

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.sync()
        return self.get_obs()

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        obs = self.get_obs()
        done = None
        reward = self.reward_func(self.data)
        self.sync()
        for _ in range(10):  # Log the data point 10 times
            self.collected_data.append((obs, self._onehot_encode_action(action)))
        return obs, reward, done, {}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        while self.viewer.is_running():
            if not self.paused:
                if self.action is not None:
                    obs, reward, done, _ = self.step(self.action)
                    print('hi')
                    self.action = None
                    if done:
                        break
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def sync(self):
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.sync()

    def get_obs(self):
        joint_state = torch.tensor(self.data.qpos[:7], dtype=torch.float32)
        end_effector_pos = torch.tensor(self.data.body("hand").xpos, dtype=torch.float32)
        target_pos = torch.tensor(self.data.body("block").xpos, dtype=torch.float32)
        input_state = torch.cat([joint_state, end_effector_pos, target_pos])
        return input_state

    def _apply_action(self, action):
        self.data.ctrl[action % 7] += 0.05 if action > 6 else -0.05
    def _onehot_encode_action(self, action):
        onehot_action = torch.zeros(14)
        onehot_action[action] = 1.0
        return onehot_action

    def key_callback(self, keycode):
        print(chr(keycode))
        if chr(keycode) == ' ':
            self.paused = not self.paused
        elif chr(keycode) == '0': self.action = 0
        elif chr(keycode) == '1': self.action = 1
        elif chr(keycode) == '3': self.action = 2
        elif chr(keycode) == '4': self.action = 3
        elif chr(keycode) == '5': self.action = 4
        elif chr(keycode) == '6': self.action = 5
        elif chr(keycode) == '7': self.action = 6
        elif chr(keycode) == '8': self.action = 7
        elif chr(keycode) == '9': self.action = 8
        elif chr(keycode) == '0': self.action = 9
        elif chr(keycode) == '-': self.action = 10
        elif chr(keycode) == '=': self.action = 11
        elif chr(keycode) == '[': self.action = 12
        elif chr(keycode) == ']': self.action = 13
        elif chr(keycode) == '.': self._dump_data()
        print(self.action)

    def _dump_data(self):
        np.save(self.file_name, np.array(self.collected_data, dtype=object))
        self.collected_data = []

    def collect_data(self, dump_folder="./data"):
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        data_point_num = 0
        self.file_name = dump_folder + f"/panda_{data_point_num}.npy"

        self.render()

        while self.viewer is not None:
            self.render()
            action = self._keyboard_input()
            if action is not None:
                obs, reward, done, _ = self.step(action)
                if len(self.collected_data) == 0:
                    self.file_name = dump_folder + f"/panda_{data_point_num}.npy"
                    data_point_num += 1
                if done:
                    break
        
    
if __name__ == "__main__":
    def calculate_reward(data):
        hand_pos = torch.tensor(data.body("hand").xpos)
        block_pos = torch.tensor(data.body("block").xpos)
        return -1 * torch.sum((hand_pos - block_pos))

    env = MujocoTeleopEnv(model_path="./mujoco_mengaerie/franka_emika_panda/scene.xml", reward_func=calculate_reward)

    env.collect_data(dump_folder="./data")

