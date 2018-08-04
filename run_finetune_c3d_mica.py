from __future__ import print_function
import sys
import os
import config_c3d
from config.config_mica import *
from Model.C3D import *
from Model.MICASplitFile import *
from Command.ComputeVolumeMean import *
from Command.Finetune import *
from Command.CreateListPrefix import *

def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(depth_path, name)
	return name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1
def append_order_to_mica_split_file(split_file):	
	new_name = []
	for name, order in zip(split_file.name, split_file.segment_order):
		new_name.append(os.path.join(name, str(int(order) - 1)))
	return new_name
c3d = C3D(
	root_folder=config_c3d.c3d_root, 
	c3d_mode=C3D_Mode.FINE_TUNING,
	pre_trained="/home/minhkv/pre-trained/depth/conv3d_mica_depth_train_iter_2000.solverstate",
	mean_file=mean_file,
	model_config=model_train,
	solver_config=solver,
	use_image=True)

train_file = MICASplitFile(
	mica_split_syntax, 
	path=mica_train_path,
	chunk_list_syntax=config_c3d.input_chunk_list_line_syntax,
	chunk_list_file=config_c3d.input_chunk_file,
	use_image=False)

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
finetune = Finetune(c3d)

c3d.generate_prototxt()
createList.execute()
# compute_mean.execute()
finetune.execute()
