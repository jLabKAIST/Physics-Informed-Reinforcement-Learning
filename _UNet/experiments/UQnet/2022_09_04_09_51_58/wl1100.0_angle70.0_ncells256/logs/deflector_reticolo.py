import gym
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from pathlib import Path
import _pickle as json
import os

class CustomEnv(gym.Env):

    #initialization
    def __init__(self, n_cells, wavelength, desired_angle):
        super(CustomEnv, self).__init__()
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(r'C:\Users\user\Chaejin Park\Physics Informed RL\UNet\reticolo_allege'));
        self.eng.addpath(self.eng.genpath('solvers'));
        os.makedirs('UNet/data',exist_ok=True)
        self.eff_file_path = 'UNet/data/'+str(wavelength)+'_'+str(desired_angle)+'_'+str(n_cells)+'_eff_table.json'
        if Path(self.eff_file_path).exists():
            with open(self.eff_file_path, 'rb') as f:
                self.eff_table = json.load(f)
        else:
            self.eff_table = {}
        self.n_cells = n_cells
        self.wavelength = matlab.double([wavelength])
        self.desired_angle = matlab.double([desired_angle])
        self.struct = np.ones(self.n_cells)
        self.eff =0

    def eval_eff_1d(self, struct, wavelength, desired_angle):
        effs = self.eng.Eval_Eff_1D(struct, wavelength, desired_angle)
        return effs

    def step(self, action): #array: input vector, ndarray
        done = False
        result_before = self.eff
        struct_after= self.struct.copy()
        
        if (struct_after[action] == 1):
            struct_after[action] = -1
        elif(struct_after[action] == -1):
            struct_after[action] = 1
        else:
            raise ValueError('action number cannot exceed cell number')
        key = tuple(struct_after.tolist())
        
        if key in self.eff_table:
            self.eff = self.eff_table[key]
        else:
            self.eff = self.eval_eff_1d(matlab.double(struct_after.tolist()), self.wavelength,\
                                             self.desired_angle)
            self.eff_table[key] = self.eff
       
        #reward = (self.eff)**3
        #various kinds of reward can be set
        #reward = (result_after)**3.
        reward = self.eff - result_before #reward = result_after - result_before
        #reward = 1-(1-result_after)**3

        self.struct = struct_after.copy()

        return struct_after.squeeze(), self.eff, reward, done

    def reset(self): #initializing the env
        self.struct = np.ones(self.n_cells)
        eff_init = 0
        self.done = False
        if self.eff_table:
	        with open(self.eff_file_path, 'wb') as f:
	            json.dump(self.eff_table, f)
        return self.struct.squeeze(), eff_init

    def get_obs(self):
        return tuple(self.struct)

    def render(self, mode= 'human', close = False):
        plt.plot(self.struct)
