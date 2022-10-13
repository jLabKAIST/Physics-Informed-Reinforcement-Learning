import numpy as np
import gym

from pirl.envs.utils import ga_init


class DeflectorEnv(gym.Env):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70
    ):
        self.n_cells = n_cells
        self.wavelength = wavelength
        self.desired_angle = desired_angle
        self.eff = None # uninitialized

        self.max_eff = -1
        self.best_struct = None

    def get_efficiency(self, struct: np.array) -> float:
        raise NotImplementedError


    def flip(self, struct, pos):
        if 0 <= pos <= (self.n_cells - 1):
            struct[pos] = 1 if struct[pos] == -1 else -1
        else:
            # if out of boundary, do nothing
            # the agent will learn the boundary
            pass

        return struct

    """
    usually, below logics will suffice
    """

    def reset(self):  # initializing the env
        self.struct = ga_init()
        self.eff = self.get_efficiency(self.struct)

        return self.struct[np.newaxis, :]  # for 1 channel

    def step(self, action):
        prev_eff = self.eff

        self.struct = self.flip(self.struct, action)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        if self.eff > self.max_eff:
            self.max_eff = self.eff
            self.best_struct = self.struct

        # unsqueeze for 1 channel
        return self.struct[np.newaxis, :], reward, False, {}