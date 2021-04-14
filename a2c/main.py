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
        self.env = CartPole()
        self.master_agent = MasterAgent(self.env.get_n_actions())
        self.agent_name = "a2c"

        ## https://stackoverflow.com/questions/54338013/parallel-import-a-python-file-from-sibling-folder
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

    def simulation_reset(self):
        self.master_agent = MasterAgent(self.env.get_n_actions())
        self.reward = []
    
    def episode_reset(self):
        self.env.reset()

    def start(self):
        start = time.time()
        print('Start!')
        for simulation_num in range(NUM_SIMULATION):
            self.simulation_reset()
            #### simulation start
            self.one_simulation_start()
            #### simulation end
            self.simulation_reward.append(self.reward)
            if (simulation_num + 1) % 10 == 0:
                DataShaping.makeCsv(self.simulation_reward, 'reward', "{}_reward_{}.csv".format(self.agent_name, simulation_num + 1))
        DataShaping.makeCsv(self.simulation_reward, 'reward', "{}_reward.csv".format(self.agent_name))
        end = time.time()
        LineNotify.send_line_notify(
            LINE_NOTIFY_FLG,
            LINE_NOTIFY_TOKEN,
            LINE_NOTIFY_MSG.format(self._shape_time(end - start))
        )
        print('End!')

    def one_simulation_start(self):
        ray.init(local_mode=False)
        agents = [Agent.remote(agent_id=i) for i in range(AGENTS_NUM)]

        states = ray.get([agent.reset_env.remote() for agent in agents])
        states = np.array(states)

        for n in range(UPDATE_NUM):
            actions = self.master_agent.select(torch.from_numpy(states))
            states = ray.get([agent.step.remote(action) for action, agent in zip(actions, agents)])
        
            trajectories = ray.get([agent.get_collect_trajectory.remote() for agent in agents])

            for trajectory in trajectories:
                trajectory["R"] = [0] * TRAJECTORY_LENGTH
                value, _ = self.master_agent(np.atleast_2d(trajectory["s2"][-1]))
                R = value[0][0].numpy()
                for i in reversed(range(TRAJECTORY_LENGTH)):
                    R = trajectory["r"][i] + GAMMA * (1 - trajectories["dones"][i]) * R
                    trajectory["R"][i] = R

            self.master_agent.learning(trajectories)

            if n % TEST_PLAY_TERM == 0:
                self.test_play()

        ray.shutdown()

    def test_play(self):
        self.env.reset()
        state = self.env.get_screen()
        agent = TestAgent()
        
        total_reward = 0
        while True:
            action = agent.select(state)
            _, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                break
            else:
                state = self.env.get_screen()
        
        self.reward.append(total_reward)
        self.env.close()

if __name__ == '__main__':
    Simulation().start()
