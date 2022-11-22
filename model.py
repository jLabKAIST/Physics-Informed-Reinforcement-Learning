import numpy as np

import torch
from torch import nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

def init_params(net, val=np.sqrt(2)):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, val)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, val)
            if module.bias is not None:
                module.bias.data.zero_()

class convrelu(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.convrelu = nn.Sequential(
            nn.Conv1d(nin, nout, 3, padding='same', padding_mode='circular'),
            nn.BatchNorm1d(nout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convrelu(x)


class ShallowUQNet(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        )
        nn.Module.__init__(self)

        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = 256
        init_params(self)


        self.conv1_1 = nn.Conv1d(1, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv1_2 = convrelu(16, 16)
        self.conv1_3 = convrelu(16, 16)
        self.pool_1 = nn.MaxPool1d(2)  # non-Uniform

        self.conv2_1 = nn.Conv1d(16, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv2_2 = convrelu(32, 32)
        self.conv2_3 = convrelu(32, 32)
        self.pool_2 = nn.MaxPool1d(2)  # non-Uniform

        self.conv3_1 = nn.Conv1d(32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv3_2 = convrelu(32, 32)
        self.conv3_3 = convrelu(32, 32)
        self.pool_3 = nn.MaxPool1d(2)  # Uniform (X

        self.conv4_1 = nn.Conv1d(32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv4_2 = convrelu(32, 32)
        self.conv4_3 = convrelu(32, 32)
        self.pool_4 = nn.MaxPool1d(2)  # Uniform (X

        self.conv6_1 = nn.Conv1d(32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv6_2 = convrelu(32, 32)
        self.conv6_3 = convrelu(32, 32)
        self.upsam_6 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv8_1 = nn.Conv1d(32 + 32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv8_2 = convrelu(32, 32)
        self.conv8_3 = convrelu(32, 32)
        self.upsam_8 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv9_1 = nn.Conv1d(32 + 32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv9_2 = convrelu(32, 32)
        self.conv9_3 = convrelu(32, 32)
        self.upsam_9 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv10_1 = nn.Conv1d(32 + 32, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv10_2 = convrelu(32, 32)
        self.conv10_3 = convrelu(32, 32)
        self.upsam_10 = nn.Upsample(scale_factor=2)  # non-Uniform

        self.conv11_1 = nn.Conv1d(16 + 32, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv11_2 = convrelu(16, 16)
        self.conv11_3 = convrelu(16, 16)

        self.conv11_fin = nn.Conv1d(16, 1, 3, padding='same', bias=True, padding_mode='circular')

    def forward(self, input_dict, state, seq_lens):
        img = input_dict['obs']

        res1_1 = self.conv1_1(img)
        temp = self.conv1_2(res1_1)
        temp = self.conv1_3(temp) + res1_1
        shortcut1 = temp
        temp = self.pool_1(shortcut1)

        res2_1 = self.conv2_1(temp)
        temp = self.conv2_2(res2_1)
        temp = self.conv2_3(temp) + res2_1
        shortcut2 = temp
        temp = self.pool_2(shortcut2)

        res3_1 = self.conv3_1(temp)
        temp = self.conv3_2(res3_1)
        temp = self.conv3_3(temp) + res3_1
        shortcut3 = temp
        temp = self.pool_3(shortcut3)

        res4_1 = self.conv4_1(temp)
        temp = self.conv4_2(res4_1)
        temp = self.conv4_3(temp) + res4_1
        shortcut4 = temp
        temp = self.pool_4(shortcut4)

        res6_1 = self.conv6_1(temp)
        temp = self.conv6_2(res6_1)
        temp = self.conv6_3(temp) + res6_1
        temp = self.upsam_6(temp)
        temp = torch.cat([temp, shortcut4], dim=1)  ######

        res8_1 = self.conv8_1(temp)
        temp = self.conv8_2(res8_1)
        temp = self.conv8_3(temp) + res8_1
        temp = self.upsam_8(temp)
        temp = torch.cat([temp, shortcut3], dim=1)  ######

        res9_1 = self.conv9_1(temp)
        temp = self.conv9_2(res9_1)
        temp = self.conv9_3(temp) + res9_1
        temp = self.upsam_9(temp)
        temp = torch.cat([temp, shortcut2], dim=1)  ######

        res10_1 = self.conv10_1(temp)
        temp = self.conv10_2(res10_1)
        temp = self.conv10_3(temp) + res10_1
        temp = self.upsam_10(temp)
        temp = torch.cat([temp, shortcut1], dim=1)  ######

        res11_1 = self.conv11_1(temp)
        temp = self.conv11_2(res11_1)
        temp = self.conv11_3(temp) + res11_1
        temp = self.conv11_fin(temp)
        
        temp = temp.flatten(1)

        return temp, []