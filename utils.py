import random
import numpy as np
import gym
import torch

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 


class OneHot(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0., high=1.,
            shape=(256, ), #### TODO fix shape
            dtype=np.float64
        )
        
    def observation(self, obs):
        obs[obs == -1] = 0

        return obs