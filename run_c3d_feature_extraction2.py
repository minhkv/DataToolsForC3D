from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
sys.path.extend(["Model", "Command"])
import config
import instance
from module_model import *
from module_command import *

c3d = instance.c3d_feature_extraction_ucf101
c3d.generate_prototxt()

train_file = instance.train_file
test_file = instance.test_file
out_file = instance.out_file_empty

train_file.clip_size = 60
out_file.clip_size = 60
train_file.load_name_and_label()
test_file.load_name_and_label()
train_file.concatenate(test_file)


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

print("Counting frame")
train_file.count_frame()
out_file.concatenate(train_file)

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=out_file, 
	use_image=False
	)
create_output_folder = CreateFeatureFolder(out_file)
feature_extraction = FeatureExtraction(c3d)

createList.execute()
create_output_folder.execute()
feature_extraction.execute()