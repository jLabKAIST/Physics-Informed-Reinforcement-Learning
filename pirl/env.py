import os
import time
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np

try:
    import matlab.engine
except:
    pass
    # raise Warning('matlab python API not installed')

RETICOLO_MATLAB = os.path.join(Path().absolute(), 'third_party/reticolo_allege')
SOLVER_MATLAB = os.path.join(Path().absolute(), 'third_party/solvers')


class ReticoloDeflector(gym.Env):
    def __init__(
            self,
            initial_eff=0,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            eff_table={},
            *args,
            **kwargs
    ):
        super(ReticoloDeflector, self).__init__()
        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(RETICOLO_MATLAB))
        self.eng.addpath(self.eng.genpath(SOLVER_MATLAB))

        self.initial_eff = initial_eff
        self.eff = 0
        self.n_cells = n_cells
        self.wavelength = matlab.double([wavelength])
        self.desired_angle = matlab.double([desired_angle])
        self.struct = np.ones(self.n_cells)
        self.eff_table = eff_table

        os.makedirs('runs', exist_ok=True)

    @property
    def config_str(self):
        return f"{self.wavelength}_{self.desired_angle}_{self.n_cells}"

    def eval_eff_1d(self, struct, wavelength, desired_angle):
        effs = self.eng.Eval_Eff_1D(struct, wavelength, desired_angle)
        return effs

    def step(self, action):
        done = False
        prev_eff = self.eff
        struct_after = self.struct.copy()

        if struct_after[action] == 1:
            struct_after[action] = -1
        elif struct_after[action] == -1:
            struct_after[action] = 1
        else:
            raise ValueError('action number cannot exceed cell number')
        key = tuple(struct_after.tolist())

        # if key in self.eff_table:
        #     self.eff = self.eff_table[key]
        # else:
        self.eff = self.eval_eff_1d(
            matlab.double(struct_after.tolist()),
            self.wavelength,
            self.desired_angle
        )
        self.eff_table[key] = self.eff

        # reward = (self.eff)**3
        # various kinds of reward can be set
        # reward = (result_after)**3.
        reward = self.eff - prev_eff  # reward = result_after - prev_eff
        # reward = 1-(1-result_after)**3

        self.struct = struct_after.copy()

        # return self.eff
        return struct_after.squeeze(), reward, done, {'eff': self.eff}

    def reset(self):  # initializing the env
        self.struct = np.ones(self.n_cells)
        self.done = False
        # if self.eff_table:
        #     with open(self.eff_file_path, 'wb') as f:
        #         json.dump(self.eff_table, f)

        # return eff_init
        return self.struct.squeeze()

    def get_obs(self):
        return tuple(self.struct)

    def render(self, mode='human', close=False):
        plt.plot(self.struct)


class DummyEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(1, kwargs['n_cells']),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(kwargs['n_cells'])

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        time.sleep(0.01)

        return self.observation_space.sample(), 0.1, False, {}

    def render(self):
        pass
