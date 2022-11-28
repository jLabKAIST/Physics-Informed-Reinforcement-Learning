import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import gym
import torch

MB = 1024 * 1024
GB = MB * 1024

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 


class OneHot(gym.ObservationWrapper):
    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=0., high=1.,
            shape=(256, ), #### TODO fix shape
            dtype=np.float64
        )
        
    def observation(self, obs):
        obs[obs == -1] = 0

        return obs

class StructureWriter(gym.Wrapper):
    def __init__(
        self, 
        env, 
        root_dir='/mnt/8tb', 
        max_folder_size=100 * GB, # 100GB
        **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        self.root_dir = self._j(root_dir, env.unwrapped.__class__.__name__)
        
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

        folder_size = sum(file.stat().st_size for file in Path(self.root_dir).rglob('*'))
        print(
            f"""writing files to {self.root_dir} current folder size: {folder_size/GB:.3f}GB"""
        )
        self.disabled = False
        
        if folder_size > max_folder_size:
            self.disabled = True
            print(
                f"""folder size is too large: {folder_size/GB}GB > {max_folder_size/GB}GB ignoring {self.__class__}"""
            )

    def _j(self, a, b):
        return os.path.join(a, b)

    def reset(self, **kwargs):
        """
        when episode is done,
        the structure(metasurface) is written to the file
        
        e.g. /mnt/8tb/MeentIndex/88-312342_20221111-123012.npy
        """
        
        obs = self.env.reset(**kwargs)
        
        if not self.disabled:
            filename = f'{self.eff*100:.6f}'.replace('.', '-')
            filename += '_' + datetime.now().strftime('%Y%m%d-%H%M%S')
            filename = self._j(self.root_dir, filename)
            np.save(filename, obs[0]) # remove channel dimenstion used for convolution
        
        return obs