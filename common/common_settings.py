import sys
import os
sys.path.append(os.path.abspath(".."))
import secret

#### Line notify ####
LINE_NOTIFY_FLG = True
LINE_NOTIFY_MSG = "実行完了\n経過時間: {}"
LINE_NOTIFY_TOKEN = secret.LINE_NOTIFY_TOKEN if LINE_NOTIFY_FLG else None

#### Simulation parameters ####
NUM_EPISODE = 2000
NUM_SIMULATION = 1

#### Replay Buffer parameters ####
NUM_REPLAY_BUFFER = 10000
RESIZE_SCREEN_SIZE_HEIGHT = 84
RESIZE_SCREEN_SIZE_WIDTH = 84
FRAME_NUM = 4
