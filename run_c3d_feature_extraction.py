from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
import config

from Model.C3D import *
from Model.UCFSplitFile import *
from Command.CreateListPrefix import *
from Command.CreateFeatureFolder import *
from Command.FeatureExtraction import *

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.FEATURE_EXTRACTION_UCF101,
	model_config=config.model_config,
	pre_trained=config.pretrained,
	mean_file=config.mean_file,
	use_image=config.use_image)

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.train_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=config.use_image,
	type_image="png")
test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	config.test_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=config.use_image,
	type_image="png")
out_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.empty_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.output_chunk_list_line_syntax,
	chunk_list_file=config.output_chunk_file,
	use_image=config.use_image,
	type_image="png")

train_file.load_name_and_label()
test_file.load_name_and_label()
train_file.concatenate(test_file)


print("Loaded: {} label".format(len(train_file.name)))

def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	return os.path.join(config.input_folder_prefix, name + '.avi')
def add_out_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.output_feature_folder, "bin", os.path.basename(video_name))
def subtract_label(label):
	return int(label) - 1
def dummy_label(label):
	return 0

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(dummy_label)

print("Counting frame")
train_file.count_frame()
out_file.concatenate(train_file)
out_file.preprocess_name(add_out_folder_prefix)

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=out_file
	)
create_output_folder = CreateFeatureFolder(out_file)
feature_extraction = FeatureExtraction(c3d)

# c3d.generate_prototxt()
# createList.execute()
# create_output_folder.execute()
# feature_extraction.execute()
