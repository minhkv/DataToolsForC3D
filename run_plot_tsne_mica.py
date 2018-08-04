from __future__ import print_function
import sys
import os
import copy

import config_c3d
# from config.config_hmdb51 import *
from config.config_mica import *
from Model.UCFSplitFile import *
from Model.MICASplitFile import *
from Model.ClassifierUsingProb import *
from Command.Classify import *
from Command.VisualizeFeature import *

def add_output_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(flow_feature_folder, config_c3d.type_feature_file, name)
	return name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1

def append_order_to_mica_split_file(split_file):	
	new_name = []
	for name, order in zip(split_file.name, split_file.segment_order):
		new_name.append(os.path.join(name, str(int(order))))
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

train_file = MICASplitFile(
	mica_split_syntax, 
	path=mica_train_path,
	# path=sample_train,
	use_image=False)

test_file = MICASplitFile(
	mica_split_syntax, 
	path=mica_test_path,
	# path=sample_test,
	use_image=False)
classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    "/home/minhkv/datasets/Kinect_vis_annotate/ActionIndex.txt"
	# "/home/minhkv/datasets/Kinect_vis_annotate/ClassTree.txt"
	# "/home/minhkv/datasets/Kinect_vis_annotate/ClassRoot.txt"
    )
mapping = UCFSplitFile(
	r"(?P<label>.+) (?P<name>.+)", 
	file_mapping
)

# print("[Load] Loading feature from: \n {}".format(config_c3d.output_feature_folder))
classInd.load_name_and_label()
train_file.load_name_and_label()
test_file.load_name_and_label()
mapping.load_name_and_label()

print("Loaded: {} train label".format(len(train_file.name)))
print("Loaded: {} test label".format(len(test_file.name)))

train_file.preprocess_name(add_output_folder_prefix)
train_file.preprocess_label(convert_label_to_int_and_subtract)

test_file.preprocess_name(add_output_folder_prefix)
test_file.preprocess_label(convert_label_to_int_and_subtract)

# train_file.preprocess_label(convert_leaf_to_group)
# test_file.preprocess_label(convert_leaf_to_group)

# train_file.preprocess_label(convert_leaf_to_root)
# test_file.preprocess_label(convert_leaf_to_root)

train_file.name = append_order_to_mica_split_file(train_file)
test_file.name = append_order_to_mica_split_file(test_file)

classifier = Classifier(
	train_file=train_file, 
	test_file=test_file, 
	class_ind=classInd, 
	classifier=config_c3d.clf,
	name=config_c3d.classifier_name,
	layer=config_c3d.layer,
	type_feature_file=config_c3d.type_feature_file
	)

plot_tsne = VisualizeFeature(classifier)
plot_tsne.execute()
