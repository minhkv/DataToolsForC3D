from __future__ import print_function
import sys
import os
import copy
import config
import instance


def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.input_folder_prefix, name, "Kinect_3", "color.avi")
	return name

def add_output_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.output_feature_folder, config.type_feature_file, name)
	return name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1

def append_order_to_mica_split_file(split_file):	
	new_name = []
	for name, order in zip(split_file.name, split_file.segment_order):
		new_name.append(os.path.join(name, order))
	return new_name
def check_empty(list_path):

	for path in list_path:
		if os.path.exists(path):
			l = os.listdir(path)
		else:
			print('[Not exist] {}'.format(path))
			continue
		if len(l) == 0:
			print("[Empty] {}".format(path))

def convert_leaf_to_group(label):
	mapping = {
		8: 0,
		9: 0,
		10: 0,
		11: 0,
		19: 1,
		20: 1,
		15: 2,
		16: 2,
		1: 3,
		2: 3,
		7: 3,
		12: 3,
		14: 3,
		4: 4,
		5: 4,
		6: 4,
		3: 5,
		13: 5,
		17: 5,
		18: 5
	}
	return mapping[label + 1]

def convert_leaf_to_root(label):
	mapping = {
		8: 0,
		9: 0,
		10: 0,
		11: 0,
		19: 0,
		20: 0,
		15: 0,
		16: 0,
		1: 1,
		2: 1,
		7: 1,
		12: 1,
		14: 1,
		4: 1,
		5: 1,
		6: 1,
		3: 1,
		13: 1,
		17: 1,
		18: 1
	}
	return mapping[label + 1]
