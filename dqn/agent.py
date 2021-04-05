import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from network import DqnNetwork, DqnDuelingNetwork
import settings

sys.path.append(os.path.abspath(".."))
from common.replaybuffer import ReplayBuffer
from common.torch_settings import DTYPE, DEVICE, Variable

def make_network(is_dueling_network, n_actions):
    if is_dueling_network:
        value_net = DqnDuelingNetwork(n_actions).type(DTYPE).to(device=DEVICE)
        target_net = DqnDuelingNetwork(n_actions).type(DTYPE)
    else:
        value_net = DqnNetwork(n_actions).type(DTYPE).to(device=DEVICE)
        target_net = DqnNetwork(n_actions).type(DTYPE)
    
    target_net.load_state_dict(value_net.state_dict())
    target_net.eval()
    target_net.to(device=DEVICE)
    return value_net, target_net

class DqnAgent(object):
    def __init__(self, env, is_soft_update=False, is_dueling_network=False):
        self.is_soft_update = is_soft_update
        self.value_net, self.target_net = make_network(is_dueling_network, env.get_n_actions())
        self.optimizer = optim.RMSprop(self.value_net.parameters(), lr=settings.NW_LEARNING_RATE, alpha=settings.NW_ALPHA, eps=settings.NW_EPS)
        self.memory = ReplayBuffer(settings.NUM_REPLAY_BUFFER, settings.FRAME_NUM)
