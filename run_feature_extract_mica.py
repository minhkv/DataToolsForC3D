from __future__ import print_function
import sys
import os
import copy

import config_c3d
import instance
# from utils_c3d import *
from config.config_mica import *
from Model.C3D import *
from Model.MICASplitFile import *
from Model.UCFSplitFile import *
from Command.CreateListPrefix import *
from Command.CreateFeatureFolder import *
from Command.FeatureExtraction import *

def dummy_label(label):
	return 0

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

def add_output_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join("/home/minhkv/feature/mica/merged_flow", config_c3d.type_feature_file, name)
	return name

c3d = C3D(
	root_folder=config_c3d.c3d_root, 
	c3d_mode=C3D_Mode.FEATURE_EXTRACTION_UCF101,
	pre_trained="/home/minhkv/pre-trained/mica/merged_flow/conv3d_mica_flow_train_iter_10000",
	mean_file=mean_file_flow,
	model_config=model_flow_feature_extract,
	# solver_config_c3d=config_c3d.solver_config_c3d,
	use_image=True)

all_file_mica = MICASplitFile(
	mica_split_syntax, 
	path=mica_train_path,
	chunk_list_syntax=config_c3d.input_chunk_list_line_syntax,
	chunk_list_file=config_c3d.input_chunk_file,
	use_image=True)

test_file = MICASplitFile(
	mica_split_syntax, 
	path=mica_test_path,
	use_image=True)
classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    "/home/minhkv/datasets/Kinect_vis_annotate/ActionIndex.txt"
    )
out_file = MICASplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config_c3d.empty_split_file_path,
	clip_size=16,
	chunk_list_syntax=config_c3d.output_chunk_list_line_syntax,
	chunk_list_file=config_c3d.output_chunk_file,
	use_image=True)
classInd.load_name_and_label()
all_file_mica.load_name_and_label()
test_file.load_name_and_label()

print("Loaded: {} train label".format(len(all_file_mica.name)))
print("Loaded: {} test label".format(len(test_file.name)))
all_file_mica.concatenate(test_file)

out_file.concatenate(all_file_mica)

all_file_mica.preprocess_name(add_input_folder_prefix)
all_file_mica.name = append_order_to_mica_split_file(all_file_mica)
all_file_mica.preprocess_label(dummy_label)

out_file.preprocess_name(add_output_folder_prefix)
out_file.name = append_order_to_mica_split_file(out_file)

all_file_mica.count_frame()
out_file.num_frames = all_file_mica.num_frames

createList = CreateListPrefix(
	split_file=all_file_mica,
	output_feature_file=out_file, 
	)

create_output_folder = CreateFeatureFolder(out_file)
feature_extract = FeatureExtraction(c3d)
print(all_file_mica.name[0])
print(out_file.name[0])
# Execute
c3d.generate_prototxt()
createList.execute()
create_output_folder.execute()
feature_extract.execute()
