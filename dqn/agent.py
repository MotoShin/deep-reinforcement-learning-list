import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(".."))
from dqn.network import DqnNetwork, DqnDuelingNetwork
from dqn.settings import *
from common.replaybuffer import ReplayBuffer
from common.torch_settings import DTYPE, DEVICE, Variable
from common.policy.egreedy import Egreedy
from common.policy.greedy import Greedy

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
        self.optimizer = optim.RMSprop(self.value_net.parameters(), lr=NW_LEARNING_RATE, alpha=NW_ALPHA, eps=NW_EPS)
        self.memory = ReplayBuffer(NUM_REPLAY_BUFFER, FRAME_NUM)
        
        # 挙動方策
        self.behavior_policy = Egreedy(
            n_actions=env.get_n_actions(),
            eps_time_steps=EPS_TIMESTEPS,
            eps_end=EPS_END,
            eps_start=EPS_START)
        # 推定方策
        self.target_policy = Greedy()

    def learning(self):
        if not self.memory.can_sample(BATCH_SIZE):
            return

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(BATCH_SIZE)

        obs_batch = Variable(torch.from_numpy(obs_batch).type(DTYPE) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).long())
        rew_batch = Variable(torch.from_numpy(rew_batch))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(DTYPE) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(DTYPE)

        if torch.cuda.is_available():
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

        # Q values
        current_Q_values = self.value_net(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
        # target Q values
        next_max_Q = self.target_policy.select(self.target_net(next_obs_batch))
        next_Q_values = not_done_mask * next_max_Q
        target_Q_values = rew_batch + (GAMMA * next_Q_values)
        # Compute Bellman error
        bellman_error = target_Q_values - current_Q_values
        # Clip the bellman error between [-1, 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        d_error = clipped_bellman_error * -1.0

        # optimize
        self.optimizer.zero_grad()
        current_Q_values.backward(d_error.data)
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.value_net.state_dict())

    def _soft_update_target_network(self):
        for target_param, value_param in zip(self.target_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(setting.TAU * value_param.data + (1.0 - TAU) * target_param.data)

    def save_memory(self, state, action, reward, done):
        index = self.memory.store_frame(state)
        self.memory.store_effect(index, action, reward, done)

    def select(self, state):
        output = None
        with torch.no_grad():
            state = Variable(state).to(DEVICE)
            output = self.value_net(state)
        return self.behavior_policy.select(output)

    def get_screen_history(self):
        return self.memory.encode_recent_observation()
