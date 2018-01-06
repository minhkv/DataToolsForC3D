from __future__ import print_function
from RemoteControl import *
import sys
import os
import config_c3d
from Model.C3D import *
from Model.UCFSplitFile import *
from Command.Finetune import *
from Command.CreateListPrefix import *
from Command.ComputeVolumeMean import *

c3d = C3D(
	root_folder=config_c3d.c3d_root, 
	c3d_mode=C3D_Mode.FINE_TUNING,
	pre_trained=config_c3d.pretrained,
	mean_file=config_c3d.mean_file,
	model_config=config_c3d.model_config,
	solver_config=config_c3d.solver_config,
	use_image=config_c3d.use_image)

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config_c3d.train_split_file_path,
	chunk_list_syntax=config_c3d.input_chunk_list_line_syntax,
	chunk_list_file=config_c3d.input_chunk_file,
	use_image=config_c3d.use_image)

train_file.load_name_and_label()
print("Loaded: {} label".format(len(train_file.name)))

def add_input_folder_prefix(path):
	# return os.path.join(config_c3d.input_folder_prefix, os.path.basename(path))
	name = os.path.basename(path).split('.')[0]
	return os.path.join(config_c3d.input_folder_prefix, name)
def convert_to_int(label):
    return int(label)
def subtract_label(label):
	return int(label) - 1

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(convert_to_int)

print("Counting frame")
train_file.count_frame()

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=None, 
	)
compute_mean = ComputeVolumeMean(c3d)
finetune = Finetune(c3d)

c3d.generate_prototxt()
createList.execute()
# compute_mean.execute()
finetune.execute()
