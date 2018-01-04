from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
import config
import instance
from Model.C3D import *
from Model.UCFSplitFile import *
from Model.MICASplitFile import *
from Command.CreateListPrefix import *
from Command.TestNet import *


def add_input_folder_annotation(path):
	name = os.path.join(config.mica_annotation_path, path + ".txt")
	return name

def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.mica_video_path, name, "Kinect_3", "color.avi")
	return name
def add_output_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.output_feature_folder, config.type_feature_file, name)
	return name
def add_output_csv_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.output_feature_folder, "csv", name)
	return name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.TEST_FINE_TUNED_NET,
	pre_trained=config.pretrained_mica,
	mean_file=config.mean_file,
	model_config=config.model_finetune_mica_test,
	use_image=False)

test_file = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_test_path,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=False)

test_file.load_name_and_label()
print("Loaded: {} label".format(len(test_file.name)))

test_file.preprocess_name(add_input_folder_prefix)
test_file.preprocess_label(convert_label_to_int_and_subtract)

createList = CreateListPrefix(
	split_file=test_file,
	output_feature_file=None, 
	)
test_net = TestNet(c3d)

# c3d.generate_prototxt()
# createList.execute()
# test_net.execute()
