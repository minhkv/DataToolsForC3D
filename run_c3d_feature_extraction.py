from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.FEATURE_EXTRACTION_UCF101,
	pre_trained=config.finetuned_ucf101_split2,
	use_image=False)
c3d.generate_prototxt()

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.train_split_file_path,
	use_image=False)
train_file.load_name_and_label()
test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	config.test_split_file_path,
	use_image=False)
test_file.load_name_and_label()
train_file.concatenate(test_file)
out_file = copy.copy(train_file)
print("Loaded: {} label".format(len(train_file.name)))

def add_input_folder_prefix(path):
	return os.path.join(config.ucf101_video_folder, os.path.basename(path))
def add_out_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.output_feature_folder, "bin", os.path.basename(video_name))
def subtract_label(label):
	return int(label) - 1
def dummy_label(label):
	return 0

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(dummy_label)
out_file.preprocess_name(add_out_folder_prefix)

print("Counting frame")
train_file.count_frame()

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=out_file, 
	use_image=False
	)
# createList.execute()

create_output_folder = CreateFeatureFolder(out_file)
# create_output_folder.execute()

feature_extraction = FeatureExtraction(c3d)
# feature_extraction.execute()