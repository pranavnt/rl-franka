import mujoco
import mujoco.viewer
import torch
from .base_env import BaseEnv

class MujocoEnv(BaseEnv):
    def __init__(self, model_path, reward_func, render=False):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.reward_func = reward_func
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.sync()
        return self.get_obs()

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        obs = self.get_obs()
        reward = self.reward_func(self.data)
        done = self._is_done()
        self.sync()
        return obs, reward, done, {}

    def render(self):
        viewer = mujoco.MjViewer(self.model, self.data)
        while True:
            viewer.render()
            if viewer.is_closed():
                break

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
    
    def sync(self):
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
            self.viewer.sync()

    def get_obs(self):
        joint_state = torch.tensor(self.data.qpos[:7], dtype=torch.float32)
        hand_pos = torch.tensor(self.data.body("hand").xpos, dtype=torch.float32)
        block_pos = torch.tensor(self.data.body("block").xpos, dtype=torch.float32)
        input_state = torch.cat([hand_pos, block_pos, joint_state])
        return input_state

    def _apply_action(self, action):
        # action is a number between 0 and 13
        self.data.ctrl[action % 7] += 0.005 if action > 6 else -0.005

    def _is_done(self):
        # Implement termination condition based on the specific environment
        return False