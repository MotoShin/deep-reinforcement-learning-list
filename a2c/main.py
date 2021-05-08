import os
import sys
import random
import torch
import time
import ray
import numpy as np
from itertools import count

sys.path.append(os.path.abspath(".."))
from a2c.agent import Agent, MasterAgent, TestAgent
from a2c.settings import *
from common.data_util import DataShaping
from common.line_notify import LineNotify
from environment.cartpole import CartPole

class Simulation(object):
    def __init__(self):
        self.reward = []
        self.simulation_reward = []
        self.ave_reward = []
        self.simulation_ave_reward = []
        self.env = CartPole()
        self.env.seed(SEED)
        self.master_agent = MasterAgent(self.env.get_n_actions())
        self.agent_name = "a2c"

        ## https://stackoverflow.com/questions/54338013/parallel-import-a-python-file-from-sibling-folder
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

    def simulation_reset(self):
        self.master_agent = MasterAgent(self.env.get_n_actions())
        self.reward = []
        self.ave_reward = []
    
    def episode_reset(self):
        self.env.reset()

    def start(self):
        start = time.time()
        print('Start!')
        for simulation_num in range(NUM_SIMULATION):
            self.simulation_reset()
            #### simulation start
            self.one_simulation_start(simulation_num)
            #### simulation end
            self.simulation_reward.append(self.reward)
            self.simulation_ave_reward.append(self.ave_reward)
            if (simulation_num + 1) % 10 == 0:
                DataShaping.makeCsv(self.simulation_reward, 'reward', "{}_reward_{}.csv".format(self.agent_name, simulation_num + 1))
                DataShaping.makeCsv(self.simulation_ave_reward, 'reward', "{}_ave_reward_{}.csv".format(self.agent_name, simulation_num + 1))
        DataShaping.makeCsv(self.simulation_reward, 'reward', "{}_reward.csv".format(self.agent_name))
        DataShaping.makeCsv(self.simulation_ave_reward, 'reward', "{}_ave_reward.csv".format(self.agent_name))
        end = time.time()
        LineNotify.send_line_notify(
            LINE_NOTIFY_FLG,
            LINE_NOTIFY_TOKEN,
            LINE_NOTIFY_MSG.format(self._shape_time(end - start))
        )
        print('End!')

    def one_simulation_start(self, simulation_num):
        ray.init(local_mode=False)
        agents = [Agent.remote(agent_id=i) for i in range(AGENTS_NUM)]

        states = ray.get([agent.reset_env.remote() for agent in agents])
        states = np.array(states)

        for n in range(UPDATE_NUM):
            self._output_progress(simulation_num, n)
            for _ in range(TRAJECTORY_LENGTH + 1):
                actions = self.master_agent.select(torch.from_numpy(states).type(DTYPE) / 255.0)
                states = ray.get([agent.step.remote(action.item()) for action, agent in zip(actions, agents)])
                states = np.array(states)
        
            trajectories = ray.get([agent.get_collect_trajectory.remote() for agent in agents])

            for trajectory in trajectories:
                trajectory["R"] = [0] * TRAJECTORY_LENGTH
                inp = np.atleast_2d(trajectory["s2"][-1])
                value, _ = self.master_agent.get_netowrk_outputs(torch.from_numpy(np.array([inp])).type(DTYPE) / 255.0)
                R = value.item()
                for i in reversed(range(TRAJECTORY_LENGTH)):
                    R = trajectory["r"][i] + GAMMA * (1 - trajectory["dones"][i]) * R
                    trajectory["R"][i] = R

            self.master_agent.learning(trajectories)

            if n % TEST_PLAY_TERM == 0:
                self.test_play()
                rewards = ray.get([agent.get_episode_reward.remote() for agent in agents])
                self.ave_reward.append(float(sum(rewards)) / AGENTS_NUM)

        for agent in agents:
            agent.close.remote()
        ray.shutdown()
        self.env.close()

    def test_play(self):
        self.master_agent.save()
        self.env.reset()
        state = self.env.get_screen()
        agent = TestAgent(self.env.get_n_actions(), self.env.get_screen())
        
        total_reward = 0
        while True:
            action = agent.select(state)
            _, reward, done, _ = self.env.step(action)
            state = self.env.get_screen()
            agent.save_memory(action, reward, done, state)
            total_reward += reward

            if done:
                break

        self.reward.append(total_reward)

    def _output_progress(self, simulation_num, update_num):
        sim = simulation_num + 1
        update = update_num + 1
        late = float((simulation_num * UPDATE_NUM + update) / (NUM_SIMULATION * UPDATE_NUM))
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

if __name__ == '__main__':
    Simulation().start()
