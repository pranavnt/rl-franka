import mujoco
import torch
from .base_env import BaseEnv

class MujocoEnv(BaseEnv):
    def __init__(self, model_path, reward_func):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.reward_func = reward_func

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self.reward_func(self.data)
        done = self._is_done()
        return obs, reward, done, {}

    def render(self):
        viewer = mujoco.MjViewer(self.model, self.data)
        while True:
            viewer.render()
            if viewer.is_closed():
                break

    def close(self):
        pass

    def _get_obs(self):
        return torch.tensor(self.data.qpos, dtype=torch.float32)

    def _apply_action(self, action):
        self.data.ctrl[:] = action

    def _is_done(self):
        # Implement termination condition based on the specific environment
        return False