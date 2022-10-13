from typing import Optional, Union, Tuple

import gym
from gym.core import ObsType, ActType
import numpy as np

from JLAB.solver import JLABCode
from pirl.envs.base_env import DeflectorEnv
from pirl.envs.utils import Direction1d, ga_init


class MeentEnv(DeflectorEnv):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70
    ):
        super(MeentEnv, self).__init__(n_cells, wavelength, desired_angle)
        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(1, n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)


    def get_efficiency(self, struct):
        # struct [1, -1, 1, 1, ...]
        struct = struct[np.newaxis, np.newaxis, :]

        wls = np.array([1100])
        period = abs(wls / np.sin(self.desired_angle / 180 * np.pi))
        calc = JLABCode(
            grating_type=0,
            n_I=1.45, n_II=1., theta=0, phi=0.,
            fourier_order=40, period=period,
            wls=wls, pol=1,
            patterns=None, ucell=struct, thickness=np.array([325])
        )

        eff, _, _ = calc.reproduce_acs_cell('p_si__real', 1)

        return eff


class DirectionEnv(MeentEnv):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            initial_pos='random', # initial agent's position
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(1, n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(len(Direction1d))
        self.initial_pos = initial_pos

    def reset(self):
        # initialize structure
        self.struct = ga_init(self.n_cells)
        self.eff = self.get_efficiency(self.struct)

        # initialize agent
        if self.initial_pos == 'center':
            self.pos = self.n_cells // 2
        elif self.initial_pos == 'right_edge':
            self.pos = self.n_cells - 1
        elif self.initial_pos == 'left_edge':
            self.pos = 0
        elif self.initial_pos == 'random':
            self.pos = np.random.randint(self.n_cells)
        else:
            raise RuntimeError('Undefined inital position')

        return self.struct[np.newaxis, :]

    def step(self, ac):
        prev_eff = self.eff
        # left == -1, noop == 0, right == 1
        # this way we can directly use ac as index difference
        ac -= 1
        self.struct = self.flip(self.struct, self.pos + ac)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        return self.struct[np.newaxis, :], reward, False, {}

