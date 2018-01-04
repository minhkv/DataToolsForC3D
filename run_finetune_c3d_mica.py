from __future__ import print_function
import sys
import os
import config
from Model.C3D import *
from Model.MICASplitFile import *
from Command.ComputeVolumeMean import *
from Command.Finetune import *
from Command.CreateListPrefix import *

def add_input_folder_annotation(path):
	name = os.path.join(config.mica_annotation_path, path + ".txt")
	return name

def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.input_folder_prefix, name, "Kinect_3", "color.avi")
	return name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.FINE_TUNING,
	pre_trained=config.pretrained,
	mean_file=config.mean_file,
	model_config=config.model_config,
	solver_config=config.solver_config,
	use_image=False)

train_file = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_train_path,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=False)

train_file.load_name_and_label()
print("Loaded: {} label".format(len(train_file.name)))

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(convert_label_to_int_and_subtract)

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=None, 
	)
compute_mean = ComputeVolumeMean(c3d)
finetune = Finetune(c3d)

# c3d.generate_prototxt()
# createList.execute()
# compute_mean.execute()
# finetune.execute()
