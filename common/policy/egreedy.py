import sys
import os
import torch
import random
import math

sys.path.append(os.path.abspath("../../"))
from common.policy.greedy import Greedy
from common.policy.schedule import LinearSchedule


class Egreedy(object):
    def __init__(self, n_actions, eps_time_steps, eps_end, eps_start):
        self.select_count = 0
        self.n_actions = n_actions
        self.decrease_schedule = LinearSchedule(eps_time_steps, eps_end, eps_start)

    def select(self, lst: torch.Tensor):
        sample = random.random()
        selected = None
        value = None

        value = self.decrease_schedule.value(self.select_count)

        if sample > value:
            selected = Greedy().select(lst)
        else:
            selected = random.randrange(self.n_actions)
        
        self.select_count += 1
        return selected
