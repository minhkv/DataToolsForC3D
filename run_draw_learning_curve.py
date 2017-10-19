from __future__ import print_function
from RemoteControl import *
import sys
import os
sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

log_files = [
    "train_split_2.log"
]
c3d = C3D(
	root_folder="/home/minhkv/C3D/C3D-v1.0/", 
	c3d_mode=C3D_Mode.DRAW_LEARNING_CURVE,
    chart_type=C3DChartType.TRAIN_LOSS_VS_ITERS,
    log_files=log_files,
	use_image=False)
draw_curve = DrawLearningCurve(c3d)
draw_curve.execute()