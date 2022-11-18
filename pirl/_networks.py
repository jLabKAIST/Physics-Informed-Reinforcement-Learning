import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import argrelextrema  ##9/19 chaejin


class convrelu(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.convrelu = nn.Sequential(nn.Conv1d(nin, nout, 3, padding='same', padding_mode='circular'),
                                      nn.BatchNorm1d(nout),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self, x):
        return self.convrelu(x)


def train(model, train_loader, optimizer):
    model.train()
    dataloss_batch = []
    criterion = nn.MSELoss()
    for idx, (img, grad) in enumerate(train_loader):
        img, grad = img.to(device), grad.to(device)
        optimizer.zero_grad()
        predictions = model(img)  #####img.permute(0, 1, 2): batch-height-width-channel
        dataloss = criterion(predictions, grad)
        dataloss_batch.append(dataloss.item())
        dataloss.backward()
        optimizer.step()
    A = np.array(dataloss_batch)
    # running_loss = sum(dataloss_batch)/len(dataloss_batch)
    running_loss = np.sqrt(np.mean(A))
    # print('train loss epoch: {:.8f}'.format(math.sqrt(running_loss)))
    return running_loss


def test(model, test_loader):
    model.eval()
    testloss_batch = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for idx, (img, grad) in enumerate(test_loader):
            img, grad = img.to(device), grad.to(device)
            outputs = model(img)
            testloss_batch.append(criterion(outputs, grad).item())
    B = np.array(testloss_batch)
    # test_loss = sum(testloss_batch)/len(testloss_batch)
    test_loss = np.sqrt(np.mean(B))
    return test_loss


def sample_action(self, obs, epsilon):
    obs = torch.reshape(obs, (1, self.nin))
    # print(obs.shape)
    out = self.forward(obs)
    coin = random.random()  # 0<coin<1
    if coin < epsilon:
        return np.random.randint(0, self.ncells)
    else:
        return out.argmax().item()


class Qnet(nn.Module):
    def __init__(self, ncells):
        super(Qnet, self).__init__()
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
        self.fc1 = nn.Linear(ncells, 2 * ncells)
        self.fc2 = nn.Linear(2 * ncells, 2 * ncells)
        self.fc3 = nn.Linear(2 * ncells, ncells)
        self.m = nn.LeakyReLU(0.1)
        init_params(self)

    def forward(self, x):
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        # print(obs.shape)
        out = self.forward(obs)
        coin = random.random()  # 0<coin<1
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            return out.argmax().item()


class UQnet(nn.Module):
    def __init__(self, ncells):
        super(UQnet, self).__init__()
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
        init_params(self)

        self.conv1_1 = nn.Conv1d(1, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv1_2 = convrelu(16, 16)
        self.conv1_3 = convrelu(16, 16)
        self.conv1_4 = convrelu(16, 16)
        self.conv1_5 = convrelu(16, 16)
        self.conv1_6 = convrelu(16, 16)
        self.pool_1 = nn.MaxPool1d(2)  # non-Uniform

        self.conv2_1 = nn.Conv1d(16, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv2_2 = convrelu(32, 32)
        self.conv2_3 = convrelu(32, 32)
        self.conv2_4 = convrelu(32, 32)
        self.conv2_5 = convrelu(32, 32)
        self.conv2_6 = convrelu(32, 32)
        self.pool_2 = nn.MaxPool1d(2)  # non-Uniform

        self.conv3_1 = nn.Conv1d(32, 64, 3, padding='same', bias=True, padding_mode='circular')
        self.conv3_2 = convrelu(64, 64)
        self.conv3_3 = convrelu(64, 64)
        self.conv3_4 = convrelu(64, 64)
        self.conv3_5 = convrelu(64, 64)
        self.conv3_6 = convrelu(64, 64)
        self.pool_3 = nn.MaxPool1d(2)  # Uniform (X

        self.conv4_1 = nn.Conv1d(64, 128, 3, padding='same', bias=True, padding_mode='circular')
        self.conv4_2 = convrelu(128, 128)
        self.conv4_3 = convrelu(128, 128)
        self.conv4_4 = convrelu(128, 128)
        self.conv4_5 = convrelu(128, 128)
        self.conv4_6 = convrelu(128, 128)
        self.pool_4 = nn.MaxPool1d(2)  # Uniform (X

        self.conv6_1 = nn.Conv1d(128, 256, 3, padding='same', bias=True, padding_mode='circular')
        self.conv6_2 = convrelu(256, 256)
        self.conv6_3 = convrelu(256, 256)
        self.conv6_4 = convrelu(256, 256)
        self.conv6_5 = convrelu(256, 256)
        self.conv6_6 = convrelu(256, 256)
        self.upsam_6 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv8_1 = nn.Conv1d(128 + 256, 128, 3, padding='same', bias=True, padding_mode='circular')
        self.conv8_2 = convrelu(128, 128)
        self.conv8_3 = convrelu(128, 128)
        self.conv8_4 = convrelu(128, 128)
        self.conv8_5 = convrelu(128, 128)
        self.conv8_6 = convrelu(128, 128)
        self.upsam_8 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv9_1 = nn.Conv1d(64 + 128, 64, 3, padding='same', bias=True, padding_mode='circular')
        self.conv9_2 = convrelu(64, 64)
        self.conv9_3 = convrelu(64, 64)
        self.conv9_4 = convrelu(64, 64)
        self.conv9_5 = convrelu(64, 64)
        self.conv9_6 = convrelu(64, 64)
        self.upsam_9 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv10_1 = nn.Conv1d(32 + 64, 32, 3, padding='same', bias=True, padding_mode='circular')
        self.conv10_2 = convrelu(32, 32)
        self.conv10_3 = convrelu(32, 32)
        self.conv10_4 = convrelu(32, 32)
        self.conv10_5 = convrelu(32, 32)
        self.conv10_6 = convrelu(32, 32)
        self.upsam_10 = nn.Upsample(scale_factor=2)  # non-Uniform

        self.conv11_1 = nn.Conv1d(16 + 32, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv11_2 = convrelu(16, 16)
        self.conv11_3 = convrelu(16, 16)
        self.conv11_4 = convrelu(16, 16)
        self.conv11_5 = convrelu(16, 16)
        self.conv11_6 = convrelu(16, 16)

        self.conv11_fin = nn.Conv1d(16, 1, 3, padding='same', bias=True, padding_mode='circular')

    def forward(self, img):
        res1_1 = self.conv1_1(img)
        temp = self.conv1_2(res1_1)
        temp = self.conv1_3(temp) + res1_1  ## sum
        res1_2 = self.conv1_4(temp)
        temp = self.conv1_5(res1_2)
        shortcut1 = self.conv1_6(temp) + res1_2
        temp = self.pool_1(shortcut1)

        res2_1 = self.conv2_1(temp)
        temp = self.conv2_2(res2_1)
        temp = self.conv2_3(temp) + res2_1
        res2_2 = self.conv2_4(temp)
        temp = self.conv2_5(res2_2)
        shortcut2 = self.conv2_6(temp) + res2_2
        temp = self.pool_2(shortcut2)

        res3_1 = self.conv3_1(temp)
        temp = self.conv3_2(res3_1)
        temp = self.conv3_3(temp) + res3_1
        res3_2 = self.conv3_4(temp)
        temp = self.conv3_5(res3_2)
        shortcut3 = self.conv3_6(temp) + res3_2
        temp = self.pool_3(shortcut3)

        res4_1 = self.conv4_1(temp)
        temp = self.conv4_2(res4_1)
        temp = self.conv4_3(temp) + res4_1
        res4_2 = self.conv4_4(temp)
        temp = self.conv4_5(res4_2)
        shortcut4 = self.conv4_6(temp) + res4_2
        temp = self.pool_4(shortcut4)

        res6_1 = self.conv6_1(temp)
        temp = self.conv6_2(res6_1)
        temp = self.conv6_3(temp) + res6_1
        res6_2 = self.conv6_4(temp)
        temp = self.conv6_5(res6_2)
        temp = self.conv6_6(temp) + res6_2
        temp = self.upsam_6(temp)
        temp = torch.cat([temp, shortcut4], dim=1)  ######

        res8_1 = self.conv8_1(temp)
        temp = self.conv8_2(res8_1)
        temp = self.conv8_3(temp) + res8_1
        res8_2 = self.conv8_4(temp)
        temp = self.conv8_5(res8_2)
        temp = self.conv8_6(temp) + res8_2
        temp = self.upsam_8(temp)
        temp = torch.cat([temp, shortcut3], dim=1)  ######

        res9_1 = self.conv9_1(temp)
        temp = self.conv9_2(res9_1)
        temp = self.conv9_3(temp) + res9_1
        res9_2 = self.conv9_4(temp)
        temp = self.conv9_5(res9_2)
        temp = self.conv9_6(temp) + res9_2
        temp = self.upsam_9(temp)
        temp = torch.cat([temp, shortcut2], dim=1)  ######

        res10_1 = self.conv10_1(temp)
        temp = self.conv10_2(res10_1)
        temp = self.conv10_3(temp) + res10_1
        res10_2 = self.conv10_4(temp)
        temp = self.conv10_5(res10_2)
        temp = self.conv10_6(temp) + res10_2
        temp = self.upsam_10(temp)
        temp = torch.cat([temp, shortcut1], dim=1)  ######

        res11_1 = self.conv11_1(temp)
        temp = self.conv11_2(res11_1)
        temp = self.conv11_3(temp) + res11_1
        res11_2 = self.conv11_4(temp)
        temp = self.conv11_5(res11_2)
        temp = self.conv11_6(temp) + res11_2
        temp = self.conv11_fin(temp)
        return temp

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        # print(obs.shape)
        out = self.forward(obs)
        coin = random.random()  # 0<coin<1
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            return out.argmax().item()


from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class ShallowUQnet(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
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
        self.lin = nn.LazyLinear(256)

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

        out = self.lin(temp)
        # temp = temp.flatten(1)
        # action_logits = temp
        # self._value_logits = temp.argmax()

        return out, []

    # def value_function(self):
    #     return self._value_logits

    # def sample_action(self, obs, epsilon):
    #     obs = torch.reshape(obs, (1, self.ncells))

    #     out = self.forward(obs)
    #     coin = random.random()  # 0<coin<1
    #     if coin < epsilon:
    #         return np.random.randint(0, self.ncells)
    #     else:
    #         return out.argmax().item()


class DuelingQnet(nn.Module):
    def __init__(self, ncells):
        super(DuelingQnet, self).__init__()
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
        self.fc1 = nn.Linear(ncells, 2 * ncells)
        self.fc2 = nn.Linear(2 * ncells, 2 * ncells)
        self.fc_a = nn.Linear(2 * ncells, ncells)
        self.fc_v = nn.Linear(2 * ncells, 1)
        self.m = nn.LeakyReLU(0.1)
        init_params(self)

    def forward(self, x):
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        a = self.fc_a(x)
        a = a - a.max(-1, keepdim=True)[0].detach()
        v = self.fc_v(x)
        q = a + v
        return q

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        # print(obs.shape)
        out = self.forward(obs)
        coin = random.random()  # 0<coin<1
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            return out.argmax().item()


def init_params(net, val=np.sqrt(2)):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, val)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, val)
            if module.bias is not None:
                module.bias.data.zero_()


def merge_network_weights(q_target_state_dict, q_state_dict, tau):
    dict_dest = dict(q_target_state_dict)
    for name, param in q_state_dict:
        if name in dict_dest:
            dict_dest[name].data.copy_((1 - tau) * dict_dest[name].data
                                       + tau * param)


def train_network(q, q_target, memory, optimizer, train_number, batch_size, gamma, double):
    # ml2
    device = 'cuda'
    for i in range(int(train_number)):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        s, a, r, s_prime, done_mask = s.to(device), a.to(device), r.to(device), s_prime.to(device), done_mask.to(
            device)  # ml2
        q_out = q(s)
        q_out = q_out.reshape(batch_size, -1)
        q_a = q_out.gather(1, a.long())  # ml2

        output_of_target = q_target(s_prime).reshape(batch_size, -1)
        if double:
            max_a_prime = q(s_prime).argmax(1, keepdim=True)

            with torch.no_grad():
                max_q_prime = output_of_target.gather(1, max_a_prime)
        else:
            with torch.no_grad():
                max_q_prime = output_of_target.max(1)[0].unsqueeze(1)

        target = done_mask * (r + gamma * max_q_prime) + (1 - done_mask) * 1 / (1 - gamma) * r
        loss = F.smooth_l1_loss(q_a, target)  # huber loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss  # for logging


class ShallowUQnet_v3(nn.Module):  # 9/15 chaejin
    def __init__(self, ncells):
        super(ShallowUQnet_v3, self).__init__()
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
        init_params(self)

        self.conv1_1 = nn.Conv1d(1, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv1_2 = convrelu(16, 16)
        self.conv1_3 = convrelu(16, 16)
        self.conv1_4 = convrelu(16, 16)
        self.conv1_5 = convrelu(16, 16)
        self.conv1_6 = convrelu(16, 16)
        self.pool_1 = nn.MaxPool1d(2)  # non-Uniform

        self.conv2_1 = nn.Conv1d(16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv2_2 = convrelu(16, 16)
        self.conv2_3 = convrelu(16, 16)
        self.conv2_4 = convrelu(16, 16)
        self.conv2_5 = convrelu(16, 16)
        self.conv2_6 = convrelu(16, 16)
        self.pool_2 = nn.MaxPool1d(2)  # non-Uniform

        self.conv3_1 = nn.Conv1d(16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv3_2 = convrelu(16, 16)
        self.conv3_3 = convrelu(16, 16)
        self.conv3_4 = convrelu(16, 16)
        self.conv3_5 = convrelu(16, 16)
        self.conv3_6 = convrelu(16, 16)
        self.pool_3 = nn.MaxPool1d(2)  # Uniform (X

        self.conv4_1 = nn.Conv1d(16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv4_2 = convrelu(16, 16)
        self.conv4_3 = convrelu(16, 16)
        self.conv4_4 = convrelu(16, 16)
        self.conv4_5 = convrelu(16, 16)
        self.conv4_6 = convrelu(16, 16)
        self.pool_4 = nn.MaxPool1d(2)  # Uniform (X

        self.conv6_1 = nn.Conv1d(16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv6_2 = convrelu(16, 16)
        self.conv6_3 = convrelu(16, 16)
        self.conv6_4 = convrelu(16, 16)
        self.conv6_5 = convrelu(16, 16)
        self.conv6_6 = convrelu(16, 16)
        self.upsam_6 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv8_1 = nn.Conv1d(16 + 16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv8_2 = convrelu(16, 16)
        self.conv8_3 = convrelu(16, 16)
        self.conv8_4 = convrelu(16, 16)
        self.conv8_5 = convrelu(16, 16)
        self.conv8_6 = convrelu(16, 16)
        self.upsam_8 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv9_1 = nn.Conv1d(16 + 16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv9_2 = convrelu(16, 16)
        self.conv9_3 = convrelu(16, 16)
        self.conv9_4 = convrelu(16, 16)
        self.conv9_5 = convrelu(16, 16)
        self.conv9_6 = convrelu(16, 16)
        self.upsam_9 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv10_1 = nn.Conv1d(16 + 16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv10_2 = convrelu(16, 16)
        self.conv10_3 = convrelu(16, 16)
        self.conv10_4 = convrelu(16, 16)
        self.conv10_5 = convrelu(16, 16)
        self.conv10_6 = convrelu(16, 16)
        self.upsam_10 = nn.Upsample(scale_factor=2)  # non-Uniform

        self.conv11_1 = nn.Conv1d(16 + 16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv11_2 = convrelu(16, 16)
        self.conv11_3 = convrelu(16, 16)
        self.conv11_4 = convrelu(16, 16)
        self.conv11_5 = convrelu(16, 16)
        self.conv11_6 = convrelu(16, 16)

        self.conv11_fin = nn.Conv1d(16, 1, 3, padding='same', bias=True, padding_mode='circular')

    def forward(self, img):
        img = img.reshape(-1, 1, self.ncells)
        res1_1 = self.conv1_1(img)
        temp = self.conv1_2(res1_1)
        temp = self.conv1_3(temp) + res1_1  ## sum
        res1_2 = self.conv1_4(temp)
        temp = self.conv1_5(res1_2)
        shortcut1 = self.conv1_6(temp) + res1_2
        temp = self.pool_1(shortcut1)

        res2_1 = self.conv2_1(temp)
        temp = self.conv2_2(res2_1)
        temp = self.conv2_3(temp) + res2_1
        res2_2 = self.conv2_4(temp)
        temp = self.conv2_5(res2_2)
        shortcut2 = self.conv2_6(temp) + res2_2
        temp = self.pool_2(shortcut2)

        res3_1 = self.conv3_1(temp)
        temp = self.conv3_2(res3_1)
        temp = self.conv3_3(temp) + res3_1
        res3_2 = self.conv3_4(temp)
        temp = self.conv3_5(res3_2)
        shortcut3 = self.conv3_6(temp) + res3_2
        temp = self.pool_3(shortcut3)

        res4_1 = self.conv4_1(temp)
        temp = self.conv4_2(res4_1)
        temp = self.conv4_3(temp) + res4_1
        res4_2 = self.conv4_4(temp)
        temp = self.conv4_5(res4_2)
        shortcut4 = self.conv4_6(temp) + res4_2
        temp = self.pool_4(shortcut4)

        res6_1 = self.conv6_1(temp)
        temp = self.conv6_2(res6_1)
        temp = self.conv6_3(temp) + res6_1
        res6_2 = self.conv6_4(temp)
        temp = self.conv6_5(res6_2)
        temp = self.conv6_6(temp) + res6_2
        temp = self.upsam_6(temp)
        temp = torch.cat([temp, shortcut4], dim=1)  ######

        res8_1 = self.conv8_1(temp)
        temp = self.conv8_2(res8_1)
        temp = self.conv8_3(temp) + res8_1
        res8_2 = self.conv8_4(temp)
        temp = self.conv8_5(res8_2)
        temp = self.conv8_6(temp) + res8_2
        temp = self.upsam_8(temp)
        temp = torch.cat([temp, shortcut3], dim=1)  ######

        res9_1 = self.conv9_1(temp)
        temp = self.conv9_2(res9_1)
        temp = self.conv9_3(temp) + res9_1
        res9_2 = self.conv9_4(temp)
        temp = self.conv9_5(res9_2)
        temp = self.conv9_6(temp) + res9_2
        temp = self.upsam_9(temp)
        temp = torch.cat([temp, shortcut2], dim=1)  ######

        res10_1 = self.conv10_1(temp)
        temp = self.conv10_2(res10_1)
        temp = self.conv10_3(temp) + res10_1
        res10_2 = self.conv10_4(temp)
        temp = self.conv10_5(res10_2)
        temp = self.conv10_6(temp) + res10_2
        temp = self.upsam_10(temp)
        temp = torch.cat([temp, shortcut1], dim=1)  ######

        res11_1 = self.conv11_1(temp)
        temp = self.conv11_2(res11_1)
        temp = self.conv11_3(temp) + res11_1
        res11_2 = self.conv11_4(temp)
        temp = self.conv11_5(res11_2)
        temp = self.conv11_6(temp) + res11_2

        temp = self.conv11_fin(temp)
        return temp

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        out = self.forward(obs)
        coin = random.random()  # 0<coin<1
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            return out.argmax().item()


class ShallowUQnet_v4(nn.Module):  # 9/14 chaejin
    def __init__(self, ncells):
        super(ShallowUQnet_v4, self).__init__()
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
        init_params(self)

        self.conv1_1 = nn.Conv1d(1, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv1_2 = convrelu(16, 16)
        self.conv1_3 = convrelu(16, 16)
        self.pool_1 = nn.MaxPool1d(2)  # non-Uniform

        self.conv2_1 = nn.Conv1d(16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv2_2 = convrelu(16, 16)
        self.conv2_3 = convrelu(16, 16)
        self.pool_2 = nn.MaxPool1d(2)  # non-Uniform

        self.conv3_1 = nn.Conv1d(16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv3_2 = convrelu(16, 16)
        self.conv3_3 = convrelu(16, 16)
        self.pool_3 = nn.MaxPool1d(2)  # Uniform (X

        self.conv4_1 = nn.Conv1d(16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv4_2 = convrelu(16, 16)
        self.conv4_3 = convrelu(16, 16)

        self.upsam_8 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv9_1 = nn.Conv1d(16 + 16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv9_2 = convrelu(16, 16)
        self.conv9_3 = convrelu(16, 16)
        self.upsam_9 = nn.Upsample(scale_factor=2)  # Uniform (X

        self.conv10_1 = nn.Conv1d(16 + 16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv10_2 = convrelu(16, 16)
        self.conv10_3 = convrelu(16, 16)
        self.upsam_10 = nn.Upsample(scale_factor=2)  # non-Uniform

        self.conv11_1 = nn.Conv1d(16 + 16, 16, 3, padding='same', bias=True, padding_mode='circular')
        self.conv11_2 = convrelu(16, 16)
        self.conv11_3 = convrelu(16, 16)

        self.conv11_fin = nn.Conv1d(16, 1, 3, padding='same', bias=True, padding_mode='circular')

    def forward(self, img):
        img = img.reshape(-1, 1, self.ncells)
        res1_1 = self.conv1_1(img)
        temp = self.conv1_2(res1_1)
        temp = self.conv1_3(temp) + res1_1  ## sum
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
        return temp

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        out = self.forward(obs)
        coin = random.random()  # 0<coin<1
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            return out.argmax().item()


class ShallowUQnet_v2(nn.Module):  # 9/14 chaejin
    def __init__(self, ncells):
        super(ShallowUQnet_v2, self).__init__()
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
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

    def forward(self, img):
        img = img.reshape(-1, 1, self.ncells)
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
        return temp

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        out = self.forward(obs)

        coin = random.random()  # 0<coin<1

        if coin < epsilon:
            new = argrelextrema(np.squeeze(out.cpu().detach().numpy()), np.greater)
            relmax = new[0]  ##type=np.array
            if relmax.size == 0:
                return out.argmax().item()
            else:
                return np.random.choice(relmax)
        else:
            return out.argmax().item()

        # 20220930 random portion flipping by chaejin
        if coin < epsilon:
            i = np.random.randint(0, self.ncells)
            j = np.random.randint(i, self.ncells)
            return (i, j)
            return np.random.randint(0, self.ncells)

        else:
            return out.argmax().item()

        ##20220920_05_22_50: mine
        ##20220920_10_04_23
        ##20220920_15_59_29
        if coin < epsilon:
            new = argrelextrema(np.squeeze(out.cpu().detach().numpy()), np.greater)
            relmax = new[0]  ##type=np.array
            if relmax.size == 0:
                return out.argmax().item()
            else:
                return np.random.choice(relmax)
        else:
            return out.argmax().item()

        # idea.1 chaejin
        new = argrelextrema(np.squeeze(out.cpu().detach().numpy()), np.greater)
        relmax = new[0]  ##type=np.array
        if relmax.size == 0:
            return out.argmax().item()
        else:
            return np.random.choice(relmax)

        # idea.2 chaejin
        if coin < epsilon:
            new = argrelextrema(np.squeeze(out.cpu().detach().numpy()), np.greater)
            relmax = new[0]  ##type=np.array
            if relmax.size == 0:
                return np.random.randint(0, self.ncells)
            else:
                return np.random.choice(relmax)
        else:
            return out.argmax().item()

        # idea.3 chaejin
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            new = argrelextrema(np.squeeze(out.cpu().detach().numpy()), np.greater)
            relmax = new[0]  ##type=np.array
            if relmax.size == 0:
                return out.argmax().item()
            elif coin < 0.8:
                return out.argmax().item()
            else:
                return np.random.choice(relmax)

        ##20220919_13_41_29: samnum's recomand - argmax 100% 신뢰
        return out.argmax().item()

        ##20220919_17_27_27: mine - argmax 에서 점점 local max로
        if coin < epsilon:
            return out.argmax().item()
        else:
            new = argrelextrema(np.squeeze(out.cpu().detach().numpy()), np.greater)
            relmax = new[0]  ##type=np.array
            if relmax.size == 0:
                return out.argmax().item()
            else:
                return np.random.choice(relmax)
