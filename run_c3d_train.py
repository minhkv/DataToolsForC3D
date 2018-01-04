from __future__ import print_function
from RemoteControl import *
import sys
import os
import config
from Model.C3D import *
from Model.UCFSplitFile import *
from Command.CreateListPrefix import *
from Command.Train import *

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.TRAINING,
	pre_trained=None,
	mean_file=config.mean_file,
    solver_config=config.solver_config,
	use_image=False)

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.train_split_file_path,
	use_image=False)


train_file.load_name_and_label()
print("Loaded: {} label".format(len(train_file.name)))

def add_input_folder_prefix(path):
	return os.path.join(config.ucf101_video_folder, os.path.basename(path))
def subtract_label(label):
	return int(label) - 1

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(subtract_label)

print("Counting frame")
train_file.count_frame()

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=None, 
	)
train = Train(c3d)

# c3d.generate_prototxt()
# createList.execute()
# train.execute()