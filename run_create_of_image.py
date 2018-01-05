from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
import instance
import config_c3d
from Command.CreateFeatureFolder import *
from Command.CreateOFImage import *
from Command.TestOFImage import *


train_file = instance.train_file
test_file = instance.test_file
out_file = copy.copy(instance.out_file_empty)
u_file = copy.copy(instance.out_file_empty)
v_file = copy.copy(instance.out_file_empty)

train_file.load_name_and_label()
test_file.load_name_and_label()
train_file.concatenate(test_file)


print("Loaded: {} label".format(len(train_file.name)))
def replace_wrong_name(name):
	wrong_name = "HandStandPushups"
	right_name = "HandstandPushups"
	new_name = name
	if wrong_name in name:
		new_name = name.replace(wrong_name, right_name)
	return new_name
def add_u_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = replace_wrong_name(name)
	return os.path.join(config_c3d.input_folder_prefix, 'u', name)
def add_v_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = replace_wrong_name(name)
	return os.path.join(config_c3d.input_folder_prefix, 'v', name)
def add_out_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config_c3d.output_feature_folder, os.path.basename(video_name))
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
print(u_file.name[0])
print(v_file.name[0])
print(out_file.name[0])

create_output_folder = CreateFeatureFolder(out_file)
create_of_image = CreateOFImage(u_file, v_file, out_file)
test_of = TestOFImage(u_file, v_file, out_file)

create_output_folder.execute()
create_of_image.execute()
test_of.execute()