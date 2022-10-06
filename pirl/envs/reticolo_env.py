import gym
import numpy as np
from gym.core import ObsType, ActType

import os
from enum import Enum
from pathlib import Path
from pirl.envs.utils import ga_init
from typing import Optional, Union, Tuple

try:
    import matlab.engine
except:
    raise Warning('matlab python API not installed')

RETICOLO_MATLAB = os.path.join(Path().absolute().parent, 'third_party/reticolo_allege')
SOLVER_MATLAB = os.path.join(Path().absolute().parent, 'third_party/solvers')


class MatlabEnv(gym.Env):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70
    ):
        self.n_cells = n_cells
        self.wavelength = wavelength
        self.desired_angle = desired_angle

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(RETICOLO_MATLAB))
        self.eng.addpath(self.eng.genpath(SOLVER_MATLAB))
        self.wavelength_mtl = matlab.double([wavelength])
        self.desired_angle_mtl = matlab.double([desired_angle])

    def flip(self, struct, pos):
        if 0 <= pos <= (self.n_cells - 1):
            struct[pos] = 1 if struct[pos] == -1 else -1
        else:
            # if out of boundary, do nothing
            # the agent will learn the boundary
            pass

        return struct

    def eval_eff_1d(self, struct: np.array):
        return self.eng.Eval_Eff_1D(
            matlab.double(struct),
            self.wavelength_mtl,
            self.desired_angle_mtl
        )


class ReticoloEnv(MatlabEnv):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(1, n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)

        self.max_eff = -1
        self.best_struct = None

    def reset(self):  # initializing the env
        self.struct = ga_init()
        self.eff = self.eval_eff_1d(self.struct)

        return self.struct[np.newaxis, :]  # for 1 channel

    def step(self, action):
        prev_eff = self.eff

        self.struct = self.flip(self.struct, action)
        self.eff = self.eval_eff_1d(self.struct)

        reward = self.eff - prev_eff

        if self.eff > self.max_eff:
            self.max_eff = self.eff
            self.best_struct = self.struct

        # unsqueeze for 1 channel
        return self.struct[np.newaxis, :], reward, False, {}


class Direction1d(Enum):
    # Directional Action
    LEFT = 0
    NOOP = 1
    RIGHT = 2


class DirectionEnv(MatlabEnv):
    def __init__(
            self,
            initial_pos='random',
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
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
        self.struct = ga_init(self.n_cells)
        self.eval_eff_1d(self.struct)

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
        self.eff = self.eval_eff_1d(self.struct)

        reward = self.eff - prev_eff

        return self.struct[np.newaxis, :], reward, False, {}
