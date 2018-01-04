from __future__ import print_function
from RemoteControl import *
import sys
import os
import config

from Model.C3D import *
from Model.UCFSplitFile import *
from Command.ComputeVolumeMean import *
from Command.Train import *
from Command.CreateListPrefix import *

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.TRAINING,
	pre_trained=config.pretrained,
	mean_file=config.mean_file,
    solver_config=config.solver_train_ucf101,
	use_image=config.use_image
	)
c3d.generate_prototxt()

train_file = UCFSplitFile(
	syntax=r"(?P<name>.+) (?P<label>\w+)", 
	path=config.train_split_file_path,
    chunk_list_file=config.input_chunk_file,
	use_image=config.use_image,
	type_image=config.type_image)


train_file.load_name_and_label()
print("Loaded: {} label".format(len(train_file.name)))

def add_input_folder_prefix(path):
    name = os.path.basename(path).split('.')[0]
    return os.path.join(config.input_folder_prefix, name)
def subtract_label(label):
	return int(label) - 1

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(subtract_label)

print("Counting frame")
train_file.count_frame()

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=None
	)
compute_mean = ComputeVolumeMean(c3d)
train_c3d = Train(c3d)

# createList.execute()
# compute_mean.execute()
# train_c3d.execute()

