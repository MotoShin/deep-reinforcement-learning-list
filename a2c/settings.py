import sys
import os
sys.path.append(os.path.abspath(".."))
from common.common_settings import *
from common.torch_settings import *

#### a2c parameters ####
AGENTS_NUM = 4
UPDATE_NUM = 500
TRAJECTORY_LENGTH = 8
TEST_PLAY_TERM = 10

#### learning parameters ####
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 10
NET_PARAMETERS_BK_PATH = 'output/value_net_bk.pth'

#### envenronment setting ####
SEED = 0

#### Replay Buffer parameters ####
FRAME_NUM = 1
