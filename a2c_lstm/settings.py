import sys
import os
sys.path.append(os.path.abspath(".."))
from common.common_settings import *
from common.torch_settings import *

#### a2c parameters ####
AGENTS_NUM = 4
UPDATE_NUM = 100
TRAJECTORY_LENGTH = 5
TEST_PLAY_TERM = 10

#### learning parameters ####
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 10
NET_PARAMETERS_BK_PATH = 'output/value_net_bk.pth'
ENTROPY_COEF_DECREASE_FLG = False
ENTROPY_COEF_DECREASE_TERM = 2000

#### envenronment setting ####
SEED = 0

#### Replay Buffer parameters ####
FRAME_NUM = 1

#### Simulation parameters ####
NUM_SIMULATION = 1
