from __future__ import print_function
import sys
import os, cv2
import config_c3d
from config.config_mica import *
from Model.C3D import *
from Model.MICASplitFile import *
from Command.ComputeVolumeMean import *
from Command.Train import *
from Command.CreateListPrefix import *

def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(stack_flow_mica, name)
	return name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1
def append_order_to_mica_split_file(split_file):	
	new_name = []
	for name, order in zip(split_file.name, split_file.segment_order):
		new_name.append(os.path.join(name, str(int(order))))
	return new_name

c3d = C3D(
	root_folder=config_c3d.c3d_root, 
	c3d_mode=C3D_Mode.FINE_TUNING,
	pre_trained=None,
	mean_file=mean_file_flow,
	model_config=model_flow_train,
	solver_config=solver,
	use_image=config_c3d.use_image)

train_file = MICASplitFile(
	mica_split_syntax,
	mica_train_path,
	chunk_list_syntax=config_c3d.input_chunk_list_line_syntax,
	chunk_list_file=config_c3d.input_chunk_file,
	use_image=config_c3d.use_image)

train_file.load_name_and_label()
print("Loaded: {} label".format(len(train_file.name)))

train_file.preprocess_name(add_input_folder_prefix)
train_file.name = append_order_to_mica_split_file(train_file)
train_file.preprocess_label(convert_label_to_int_and_subtract)
train_file.count_frame()

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=None, 
	)
compute_mean = ComputeVolumeMean(c3d)
train = Train(c3d)


c3d.generate_prototxt()
createList.execute()
# compute_mean.execute()
train.execute()