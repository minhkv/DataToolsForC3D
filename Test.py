from __future__ import print_function
from RemoteControl import *
import sys
import os
sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

vid = Video("/home/minhkv/datasets/UCF101/v_BlowingCandles_g05_c03.avi")

num_frames, fps = vid.get_frame_count()
print (num_frames)