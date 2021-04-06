import sys
import os

sys.path.append(os.path.abspath(".."))
from common.common_settings import *

#### network parameters ####
NW_LEARNING_RATE = 0.00025
NW_ALPHA = 0.95
NW_EPS = 0.01

#### epsilon-greedy parameters ####
EPS_START = 1.0
EPS_END = 0.01
EPS_TIMESTEPS = 950
