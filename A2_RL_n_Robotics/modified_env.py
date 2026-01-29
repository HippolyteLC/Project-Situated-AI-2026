import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
import gymnasium_robotics

gymnasium_robotics.register_robotics_envs()

class FetchPickAndPlaceModifiedWrapper(Wrapper):
    def __init__(self, env, contact_bonus=0.1, lift_bonus=0.2):
        super().__init__(env)
        old_space = self.env.observation_space.spaces['observation']
        # self.observation_space.spaces['observation'] = gym.spaces.Box(
        #     low=np.append(old_space.low, 0),
        #     high=np.append(old_space.high, 1),
        #     dtype=np.float32
        # )

    def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)

            obj_pos = obs['observation'][3:6]
            gripper_pos = obs['observation'][0:3]
            goal_pos = obs['desired_goal']
            finger_width = obs['observation'][9] + obs['observation'][10]
            dist_to_obj = np.linalg.norm(obj_pos - gripper_pos)

            is_lifted = obj_pos[2] > 0.43  # Table is at 0.4
            is_gripped = finger_width < 0.02 and dist_to_obj < 0.03
            has_object = 1.0 if (is_lifted and is_gripped) else 0.0

            # bonus for getting to obj
            if dist_to_obj < 0.03:
                reward += 0.1 
            # bonus for gripping and lifting obj
            if has_object:
                reward += 0.4 

                dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
                reward += (1.0 - np.tanh(dist_to_goal)) * 0.2 #use tanh to get 1.0 if far, 0, if close. Then 
                # carrying the reward closer to the goal can give + 0.2 when very close (incrementally)

            # obs['observation'] = np.append(obs['observation'], has_object)

            return obs, reward, terminated, truncated, info

gym.register(
    id="FetchPickAndPlaceModified-v0",
    entry_point=lambda **kwargs: FetchPickAndPlaceModifiedWrapper(
        gym.make("FetchPickAndPlace-v4", reward_type="dense", **kwargs)
    ),
    max_episode_steps=50,
)
