import numpy as np
import gym


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


    def flip(self, struct, pos):
        if 0 <= pos <= (self.n_cells - 1):
            struct[pos] = 1 if struct[pos] == -1 else -1
        else:
            # if out of boundary, do nothing
            # the agent will learn the boundary
            pass

        return struct

    def get_efficiency(self, struct: np.array) -> float:
        raise NotImplementedError