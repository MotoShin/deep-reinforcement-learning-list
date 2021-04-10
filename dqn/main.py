import os
import sys
import random
import torch
import time
import numpy as np
from itertools import count

sys.path.append(os.path.abspath(".."))
from dqn.agent import DqnAgent
from dqn.settings import *
from common.data_util import DataShaping
from common.line_notify import LineNotify
from environment.cartpole import CartPole


class Simulation(object):
    def __init__(self):
        self.env = CartPole()
        self.agent = None
        self.reward = []
        self.simulation_reward = []
        self.agent_name = "dqn"

    def simulation_reset(self):
        self.env.reset()
        self.agent = DqnAgent(self.env)
        self.init_memory()
        self.reward = []
    
    def episode_reset(self):
        self.env.reset()

    def start(self):
        start = time.time()
        print("Start!")
        for simulation_num in range(NUM_SIMULATION):
            self.simulation_reset()
            #### simulation start
            self.one_simulation_start(simulation_num)
            #### simulation end
            self.simulation_reward.append(self.reward)
            if (simulation_num + 1) % 10 == 0:
                DataShaping.makeCsv(self.simulation_reward, 'reward', "{}_reward_{}.csv".format(self.agent_name, simulation_num + 1))
        DataShaping.makeCsv(self.simulation_reward, 'reward', "{}_reward.csv".format(self.agent_name))
        end = time.time()
        self.env.close()
        LineNotify.send_line_notify(
            LINE_NOTIFY_FLG,
            LINE_NOTIFY_TOKEN,
            LINE_NOTIFY_MSG.format(self._shape_time(end - start))
        )
    
    def one_simulation_start(self, simulation_num):
        for episode_num in range(NUM_EPISODE):
            self.episode_reset()
            self.one_episode_start(simulation_num=simulation_num, episode_num=episode_num)
            if episode_num % TARGET_UPDATE == 0:
                self.agent.update_target_network()

    def one_episode_start(self, simulation_num, episode_num):
        self.output_progress(simulation_num, episode_num)
        current_screen = self.env.get_screen()
        state = current_screen
        sum_reward = 0.0
        for t in count():
            # Chose action
            recent_screen = self.agent.get_screen_history()
            inp = torch.from_numpy(np.array([recent_screen])).type(DTYPE) / 255.0
            action = self.agent.select(inp)
            # Action
            _, reward, done, _ = self.env.step(action)
            screen = self.env.get_screen()
            if done:
                reward = self.env.episode_end_reward(reward)
                self.env.reset()
                screen = self.env.get_screen()
            # Move to the next state
            sum_reward += reward
            state = screen

            # Store the next state
            self.agent.save_memory(state, action, reward, done)

            # learning
            self.agent.learning()
            if done:
                self.reward.append(sum_reward)
                break

    def output_progress(self, simulation_num, episode_num):
        sim = simulation_num + 1
        epi = episode_num + 1
        late = float((simulation_num * NUM_EPISODE + epi) / (NUM_SIMULATION * NUM_EPISODE))
        late_percent = late * 100
        if (late_percent % 10 == 0):
            print("progress: {: >3} %".format(late_percent))

    def _shape_time(self, elapsed_time):
        elapsed_time = int(elapsed_time)

        elapsed_hour = elapsed_time // 3600
        elapsed_minute = (elapsed_time % 3600) // 60
        elapsed_second = (elapsed_time % 3600 % 60)

        return str(elapsed_hour).zfill(2) + ":" \
                + str(elapsed_minute).zfill(2) + ":" \
                + str(elapsed_second).zfill(2)

    def init_memory(self):
        state = self.env.get_screen()
        action = random.randrange(self.env.get_n_actions())
        _, reward, done, _ = self.env.step(action)
        self.agent.save_memory(state, action, reward, done)
        self.env.reset()

if __name__ == '__main__':
    Simulation().start()
