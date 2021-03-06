import sys
import os
sys.path.append(os.path.abspath(".."))
from common.common_settings import *
from common.torch_settings import *

#### network parameters ####
NW_LEARNING_RATE = 0.00025
NW_ALPHA = 0.95
NW_EPS = 0.01

#### epsilon-greedy parameters ####
EPS_START = 1.0
EPS_END = 0.01
EPS_TIMESTEPS = 950

#### learning parameters ####
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10
NET_PARAMETERS_BK_PATH = 'output/value_net_bk.pth'
# Soft Update Setting
TAU = 1e-3
