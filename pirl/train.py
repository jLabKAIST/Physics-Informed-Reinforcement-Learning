import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


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
            dict_dest[name].data.copy_(
                (1 - tau) * dict_dest[name].data + tau * param
            )


def train_network(q, q_target, memory, optimizer, train_number, batch_size,
                  gamma, double):
    # ml2
    device = 'cuda'
    for i in range(int(train_number)):
        l = memory.sample(batch_size)
        s, a, r, s_prime, done_mask = tuple(map(lambda x: x.to(device), l))

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