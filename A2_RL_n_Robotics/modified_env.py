import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
import gymnasium_robotics

# Essential: This registers the base Fetch environments
gymnasium_robotics.register_robotics_envs()

class FetchPushModifiedWrapper(Wrapper):
    def __init__(self, env, contact_bonus=0.1):
        super().__init__(env)
        self.contact_bonus = contact_bonus
        
        old_shape = self.env.observation_space['observation'].shape[0]
        new_shape = old_shape + 3
        self.observation_space['observation'] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

    def _get_relative_dist(self, obs):
        gripper_pos = obs['observation'][0:3]
        object_pos = obs['observation'][3:6]
        return object_pos - gripper_pos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rel_dist = self._get_relative_dist(obs)
        obs['observation'] = np.concatenate([obs['observation'], rel_dist])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rel_dist = self._get_relative_dist(obs)
        obs['observation'] = np.concatenate([obs['observation'], rel_dist])
        
        if np.linalg.norm(rel_dist) < 0.05:
            reward += self.contact_bonus
            
        return obs, reward, terminated, truncated, info

gym.register(
    id="FetchPushModified-v0",
    entry_point=lambda **kwargs: FetchPushModifiedWrapper(
        gym.make("FetchPush-v4", reward_type="dense", **kwargs)
    ),
    max_episode_steps=50,
)
