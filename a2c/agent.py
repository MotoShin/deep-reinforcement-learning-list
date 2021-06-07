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
from a2c.memory import Memory
from a2c.settings import *
from common.torch_settings import DTYPE, DEVICE, Variable
from environment.cartpole import CartPole


@ray.remote
class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.env = CartPole()
        self.state = None
        self.memory = Memory()
        self.reward = 0
        self.episode_reward = 0

    def reset_env(self):
        self.env.reset()
        self.state = self.env.get_screen()
        return self.env.get_screen()

    def step(self, action):
        _, reward, done, _ = self.env.step(action)
        next_screen = self.env.get_screen()
        self.reward += reward

        self.memory.add(np.copy(self.state), reward, action, done, next_screen)

        if done:
            self.env.reset()
            self.episode_reward = self.reward
            self.reward = 0
            self.state = self.env.get_screen()
        else:
            self.state = self.env.get_screen()

        return self.state

    def get_collect_trajectory(self):
        # 蓄積したtrajectoryの回収
        obs_batch, act_batch, rew_batch, done_mask, next_obs_batch = self.memory.rollout()
        trajectory = {"s": obs_batch, "a": act_batch, "r": rew_batch, "s2": next_obs_batch, "dones": done_mask}
        self.memory = Memory()
        return trajectory

    def get_episode_reward(self):
        return self.episode_reward

    def close(self):
        self.env.close()

class MasterAgent:
    def __init__(self, action_num):
        self.network = ActorCriticNetwork(action_num).type(DTYPE).to(device=DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        self.entropy_coef = 0.01
        self.record_loss = []
        self.record_actor_loss = []
        self.record_critic_loss = []
        self.record_entropy = []

    def select(self, states):
        action_probs = None
        with torch.no_grad():
            states = Variable(states)
            _, action_probs = self.network(states)
        actions = action_probs.sample()

        return actions

    def get_netowrk_outputs(self, states):
        value = None
        action_probs = None
        with torch.no_grad():
            states = Variable(states)
            value, action_probs = self.network(states)
        return value, action_probs

    def save(self):
        torch.save(self.network.state_dict(), NET_PARAMETERS_BK_PATH)

    def learning(self, trajectories, count):
        if ENTROPY_COEF_DECREASE_FLG and count % ENTROPY_COEF_DECREASE_TERM == 0:
            self.entropy_coef = self.entropy_coef / 2.0 if (self.entropy_coef / 2.0) > 3.0e-10 else 3.0e-10
        (states, actions, next_states, rewards, dones, discounted_returns) = [], [], [], [], [], []
        
        for trajectory in trajectories:
            states += trajectory["s"]
            actions += trajectory["a"]
            next_states += trajectory["s2"]
            rewards += trajectory["r"]
            dones += trajectory["dones"]
            discounted_returns += trajectory["R"]

        states = Variable(torch.from_numpy(np.array(states, dtype=np.float32)).type(DTYPE) / 255.0)
        actions = Variable(torch.from_numpy(np.array(actions)).long())
        rewards = Variable(torch.from_numpy(np.array(rewards, dtype=np.float32)))
        next_states = Variable(torch.from_numpy(np.array(next_states, dtype=np.float32)).type(DTYPE) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - np.array(dones))).type(DTYPE)
        discounted_returns = Variable(torch.from_numpy(np.array(discounted_returns, dtype=np.float32)))

        if torch.cuda.is_available():
            actions = actions.cuda()
            rewards = rewards.cuda()
            discounted_returns = discounted_returns.cuda()

        self.network.train()
        values, action_probs = self.network(states)
        log_probs = action_probs.log_prob(actions)

        ary_entropy = action_probs.entropy()
        entropy = torch.mean(ary_entropy)
        
        # https://medium.com/programming-soda/advantage%E3%81%A7actor-critic%E3%82%92%E5%AD%A6%E7%BF%92%E3%81%99%E3%82%8B%E9%9A%9B%E3%81%AE%E6%B3%A8%E6%84%8F%E7%82%B9-a1b3925bc3e6
        advantage = discounted_returns - values.squeeze(1).detach()

        actor_loss = (-1 * log_probs * advantage.detach()).mean()
        critic_loss = ((values.squeeze(1) - discounted_returns).pow(2)).mean()

        loss = actor_loss + 0.5 * critic_loss + -1 * self.entropy_coef * entropy
        # print(loss.item())

        self.record_loss.append(loss.item())
        self.record_actor_loss.append(actor_loss.item())
        self.record_critic_loss.append(critic_loss.item())
        self.record_entropy.append(entropy.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_record_param(self):
        return [self.record_loss, self.record_actor_loss, self.record_critic_loss, self.record_entropy]

class TestAgent:
    def __init__(self, action_num):
        self.network = ActorCriticNetwork(action_num).type(DTYPE).to(device=DEVICE)
        self.network.load_state_dict(torch.load(NET_PARAMETERS_BK_PATH))
    
    def select(self, state):
        action_prob = None
        self.network.eval()
        with torch.no_grad():
            inp = Variable(torch.from_numpy(np.array([state], dtype=np.float32)).type(DTYPE) / 255.0).to(DEVICE)
            _, action_prob = self.network(inp)
        action = action_prob.sample()
        return action.item()
