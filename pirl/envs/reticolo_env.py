import gym
import numpy as np

import os
from pathlib import Path

from pirl.envs.base_env import DeflectorEnv
from pirl.envs.utils import ga_init, Direction1d

try:
    import matlab.engine
except:
    raise Warning('matlab python API not installed')

RETICOLO_MATLAB = os.path.join(Path().absolute().parent, 'third_party/reticolo_allege')
SOLVER_MATLAB = os.path.join(Path().absolute().parent, 'third_party/solvers')


class MatlabEnv(DeflectorEnv):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(RETICOLO_MATLAB))
        self.eng.addpath(self.eng.genpath(SOLVER_MATLAB))
        self.wavelength_mtl = matlab.double([wavelength])
        self.desired_angle_mtl = matlab.double([desired_angle])

    def get_efficiency(self, struct: np.array):
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


class DirectionEnv(MatlabEnv):
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
