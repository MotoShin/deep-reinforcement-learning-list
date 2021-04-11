import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import ray

sys.path.append(os.path.abspath(".."))
from a2c.network import ActorCriticNetwork
from a2c.settings import *
from common.torch_settings import DTYPE, DEVICE, Variable
from environment.cartpole import CartPole


@ray.remote
class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.env = CartPole()
        self.state = None
        self.memory = ReplayBuffer(TRAJECTORY_LENGTH, FRAME_NUM)

    def reset_env(self):
        self.env.reset()
        self.state = self.env.get_screen()
        return self.state

    def step(self, action):
        state = self.state
        _, reward, done, _ = self.env.step(action)

        self._save_memory(state, action, reward, done)

        if done:
            self.env.reset()
            self.state = self.env.get_screen()
        else:
            self.state = self.env.get_screen()
        
        return self.state

    def get_collect_trajectory(self):
        # 蓄積したtrajectoryの回収
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(TRAJECTORY_LENGTH)
        trajectory = {"s": obs_batch, "a": act_batch, "r": rew_batch, "s2": next_obs_batch, "dones": done_mask}
        self.memory = ReplayBuffer(TRAJECTORY_LENGTH, FRAME_NUM)
        return trajectory
    
    def _save_memory(self, state, action, reward, done):
        index = self.memory.store_frame(state)
        self.memory.store_effect(index, action, reward, done)

class MasterAgent:
    def __init__(self, action_num):
        self.network = ActorCriticNetwork(action_num)

    def select(self, states):
        action_probs = None
        with torch.no_grad():
            states = Variable(states)
            value, action_probs = self.network(states)
        dist = torch.distributions.categorical.Categorical(action_probs)
        actions = dist.sample()

        return actions

    def learning(self, trajectories):
        (states, actions, next_states, rewards, dones, discounted_returns) = [], [], [], [], [], []
        
        for trajectory in trajectories:
            states += trajectories["s"]
            actions += trajectories["a"]
            next_states += trajectories["s2"]
            rewards += trajectories["r"]
            dones += trajectories[dones]
            discounted_returns += trajectories["R"]

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
        dist = torch.distributions.categorical.Categorical(action_probs)
        actions = dist.sample()
