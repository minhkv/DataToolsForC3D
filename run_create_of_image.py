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
out_file = copy.copy(instance.out_file_empty)
u_file = copy.copy(instance.out_file_empty)
v_file = copy.copy(instance.out_file_empty)

train_file.load_name_and_label()
test_file.load_name_and_label()
train_file.concatenate(test_file)


print("Loaded: {} label".format(len(train_file.name)))

def add_u_folder_prefix(path):
	return os.path.join(config.ucf101_tvl1_flow_folder, 'u', os.path.basename(path))
def add_v_folder_prefix(path):
	return os.path.join(config.ucf101_tvl1_flow_folder, 'v', os.path.basename(path))
def add_out_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.ucf101_stack_tvl1_folder, os.path.basename(video_name))
def subtract_label(label):
	return int(label) - 1
def dummy_label(label):
	return 0
u_file.concatenate(train_file)
u_file.preprocess_name(add_u_folder_prefix)
v_file.concatenate(train_file)
v_file.preprocess_name(add_v_folder_prefix)

out_file.concatenate(train_file)
out_file.preprocess_name(add_out_folder_prefix)


create_output_folder = CreateFeatureFolder(out_file)
create_of_image = CreateOFImage(u_file, v_file, out_file)

# create_output_folder.execute()
create_of_image.execute()