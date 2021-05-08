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
        self.env.seed(SEED)
        self.state = None
        self.memory = ReplayBuffer(TRAJECTORY_LENGTH + 1, FRAME_NUM)
        self.index = 0
        self.reward = 0
        self.episode_reward = 0

    def reset_env(self):
        self.env.reset()
        state = self.env.get_screen()
        action = random.randrange(self.env.get_n_actions())
        _, reward, done, _ = self.env.step(action)
        self.index = self.memory.store_frame(state)
        return self.memory.encode_recent_observation()

    def step(self, action):
        state = self.state
        _, reward, done, _ = self.env.step(action)
        self.reward += reward

        self.memory.store_effect(self.index, action, reward, done)

        if done:
            self.env.reset()
            self.episode_reward = self.reward
            self.reward = 0
            self.state = self.env.get_screen()
        else:
            self.state = self.env.get_screen()
        
        self.index = self.memory.store_frame(self.state)
        
        return self.memory.encode_recent_observation()

    def get_collect_trajectory(self):
        # 蓄積したtrajectoryの回収
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(TRAJECTORY_LENGTH)
        trajectory = {"s": obs_batch, "a": act_batch, "r": rew_batch, "s2": next_obs_batch, "dones": done_mask}
        self.memory = ReplayBuffer(TRAJECTORY_LENGTH + 1, FRAME_NUM)
        self.index = self.memory.store_frame(self.state)
        return trajectory

    def get_episode_reward(self):
        return self.episode_reward

    def close(self):
        self.env.close()

class MasterAgent:
    def __init__(self, action_num):
        self.network = ActorCriticNetwork(action_num).type(DTYPE).to(device=DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)

    def select(self, states):
        action_probs = None
        self.network.eval()
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
            states.append(trajectory["s"])
            actions.append(trajectory["a"])
            next_states.append(trajectory["s2"])
            rewards.append(trajectory["r"])
            dones.append(trajectory["dones"])
            discounted_returns.append(trajectory["R"])

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
        values, action_probs = self.network.parallel_input(states)
        log_probs = []
        for action_prob in action_probs:
            action = action_prob.sample()
            log_probs.append(action_prob.log_prob(action))
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        ary_entropy = [action_prob.entropy() for action_prob in action_probs]
        ary_entropy = torch.stack(ary_entropy)
        ary_entropy_sum = torch.sum(ary_entropy, dim=0, keepdim=True)
        entropy = torch.mean(ary_entropy_sum)
        
        # https://medium.com/programming-soda/advantage%E3%81%A7actor-critic%E3%82%92%E5%AD%A6%E7%BF%92%E3%81%99%E3%82%8B%E9%9A%9B%E3%81%AE%E6%B3%A8%E6%84%8F%E7%82%B9-a1b3925bc3e6
        advantage = discounted_returns - values.squeeze(2).detach()

        actor_loss = (log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = -1 * actor_loss + 0.5 * critic_loss + -1 * 0.01 * entropy
        # print(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class TestAgent:
    def __init__(self, action_num, state):
        self.network = ActorCriticNetwork(action_num).type(DTYPE).to(device=DEVICE)
        self.network.load_state_dict(torch.load(NET_PARAMETERS_BK_PATH))
        self.memory = ReplayBuffer(FRAME_NUM + 1, FRAME_NUM)
        self.index = self.memory.store_frame(state)
    
    def select(self, state):
        inp = self.memory.encode_recent_observation()
        action_probs = None
        self.network.eval()
        with torch.no_grad():
            inp = Variable(torch.from_numpy(np.array([inp], dtype=np.float32)).type(DTYPE) / 255.0).to(DEVICE)
            value, action_prob = self.network(inp)
        action = action_prob.sample()
        return action.item()

    def save_memory(self, action, reward, done, next_state):
        self.memory.store_effect(self.index, action, reward, done)
        self.index = self.memory.store_frame(next_state)
