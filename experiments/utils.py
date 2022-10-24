import gym
import numpy as np
import torch
from torch import nn


def load_pretrained(path):
    pretrained = torch.load(path)

    # add dummy for rllib compat
    pretrained["advantage_module.dueling_A_0._model.0.weight"] = nn.Parameter(
        torch.from_numpy(np.random.randn(256, 256)))
    pretrained["advantage_module.dueling_A_0._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))
    pretrained["advantage_module.A._model.0.weight"] = nn.Parameter(torch.from_numpy(np.random.randn(256, 256)))
    pretrained["advantage_module.A._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))

    return pretrained

# def load_pretrained_apex(path):
#     pretrained = torch.load(path)
#
#     # add dummy for rllib compat
#     pretrained["value_module.dueling_V_0._model.0.weight"] = nn.Parameter(
#         torch.from_numpy(np.random.randn(256, 256)))
#     pretrained["value_module.dueling_V_0._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))
#     pretrained["value_module.V._model.0.weight"] = nn.Parameter(torch.from_numpy(np.random.randn(256, 256)))
#     pretrained["value_module.V._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))
#
#     return pretrained

class RegretWrapper(gym.Wrapper):
    def step(self, ac):
        obs, rew, done, info = super(RegretWrapper, self).step(ac)
        info['regret'] = 100. - rew

        return obs, rew, done, info