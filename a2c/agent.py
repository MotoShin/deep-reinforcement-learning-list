import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import ray
import random

sys.path.append(os.path.abspath(".."))
from a2c.network import ActorCriticNetwork
from a2c.settings import *
from common.torch_settings import DTYPE, DEVICE, Variable
from common.replaybuffer import ReplayBuffer
from environment.cartpole import CartPole


@ray.remote
class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.env = CartPole()
        self.state = None
        self.memory = ReplayBuffer(TRAJECTORY_LENGTH + 1, FRAME_NUM)
        self.index = 0

    def reset_env(self):
        self.env.reset()
        state = self.env.get_screen()
        action = random.randrange(self.env.get_n_actions())
        _, reward, done, _ = self.env.step(action)
        self.index = self.memory.store_frame(state)
        self.memory.store_effect(self.index, action, reward, done)
        self.env.reset()
        self.state = self.env.get_screen()
        return self.memory.encode_recent_observation()

    def step(self, action):
        state = self.state
        self.index = self.memory.store_frame(state)
        _, reward, done, _ = self.env.step(action)

        self.memory.store_effect(self.index, action, reward, done)

        if done:
            self.env.reset()
            self.state = self.env.get_screen()
        else:
            self.state = self.env.get_screen()
        
        return self.memory.encode_recent_observation()

    def get_collect_trajectory(self):
        # 蓄積したtrajectoryの回収
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(TRAJECTORY_LENGTH)
        trajectory = {"s": obs_batch, "a": act_batch, "r": rew_batch, "s2": next_obs_batch, "dones": done_mask}
        self.memory = ReplayBuffer(TRAJECTORY_LENGTH, FRAME_NUM)
        return trajectory

class MasterAgent:
    def __init__(self, action_num):
        self.network = ActorCriticNetwork(action_num)
        self.optimizer = optim.Adam(self.network.parameters())

    def select(self, states):
        action_probs = None
        with torch.no_grad():
            states = Variable(states)
            value, action_probs = self.network(states)
        actions = action_probs.sample()

        return actions

    def get_netowrk_outputs(self, states):
        return self.network(states)

    def save(self):
        torch.save(self.network.state_dict(), NET_PARAMETERS_BK_PATH)

    def learning(self, trajectories):
        (states, actions, next_states, rewards, dones, discounted_returns) = [], [], [], [], [], []
        
        for trajectory in trajectories:
            # TODO: ここがうまくいかない
            states += trajectory["s"]
            actions += trajectory["a"]
            next_states += trajectory["s2"]
            rewards += trajectory["r"]
            dones += trajectory["dones"]
            discounted_returns += trajectory["R"]

        # TODO: ここらへん要検証
        states = Variable(torch.from_numpy(np.array(states, dtype=np.float32)).type(DTYPE) / 255.0)
        actions = Variable(torch.from_numpy(np.array(actions)).long())
        rewards = Variable(torch.from_numpy(np.array(rewards, dtype=np.float32)))
        next_states = Variable(torch.from_numpy(np.array(next_states, dtype=np.float32)).type(DTYPE) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - np.array(dones))).type(DTYPE)
        discounted_returns = Variable(torch.from_numpy(np.array(discounted_returns, type=np.float32)))

        if torch.cuda.is_available():
            actions = actions.cuda()
            rewards = rewards.cuda()
            discounted_returns = discounted_returns.cuda()
        
        values, action_probs = self.network(states)
        actions = action_probs.sample()
        log_probs = action_probs.log_prob()
        
        ary_entropy = action_probs.entropy()
        entropy = 0
        for ary in ary_entropy:
            entropy += ary.mean()
        
        adavantage = discounted_returns - values

        actor_loss = -(log_probs * adavantage.detach()).mean()
        critic_loss = adavantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class TestAgent:
    def __init__(self):
        self.network = ActorCriticNetwork()
        self.network.load_state_dict(NET_PARAMETERS_BK_PATH)

    def select(self, state):
        action_probs = None
        with torch.no_grad():
            state = Variable(state)
            value, action_prob = self.network(state)
        action = action_prob.sample()
        return action
