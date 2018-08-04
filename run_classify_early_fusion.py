from __future__ import print_function
import sys
import os
import copy
import numpy as np
import config_c3d
from config.config_mica import *
from Model.UCFSplitFile import *
from Model.MICASplitFile import *
from Model.ClassifierUsingProb import *
from Command.ClassifyEarlyFusion import *
from Command.VisualizeEarlyFusion import *
import instance

def add_input_rgb_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(rgb_feature_folder, config_c3d.type_feature_file, os.path.basename(video_name))
def add_input_flow_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(flow_feature_folder, config_c3d.type_feature_file, os.path.basename(video_name))

def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1

def fuse_function(vect1, vect2):
	a = 0.5
	fused_vec = (a * np.array(vect1) + (1 - a) * np.array(vect2)) #/ 2.0
	return fused_vec
def append_order_to_mica_split_file(split_file):	
	new_name = []
	for name, order in zip(split_file.name, split_file.segment_order):
		new_name.append(os.path.join(name, str(int(order))))
	return new_name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1
def concatenate_l2_norm(vect1, vect2):
	norm_vect1 = np.linalg.norm(vect1)
	norm_vect2 = np.linalg.norm(vect2)
	vect1_unit = vect1
	vect2_unit = vect2
	if norm_vect1 != 0:
		vect1_unit = vect1 / float(np.linalg.norm(vect1))
	if norm_vect2 != 0:
		vect2_unit = vect2 / float(np.linalg.norm(vect2))
	fused_vec = np.concatenate((np.array(vect1_unit), np.array(vect2_unit)))
	return fused_vec

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
classInd.load_name_and_label()

train_file.load_name_and_label()
test_file.load_name_and_label()
print("Loaded: {} train label".format(len(train_file.name)))
print("Loaded: {} test label".format(len(test_file.name)))

train_file_flow = copy.copy(train_file)
test_file_flow = copy.copy(test_file)

#rgb
train_file.preprocess_name(add_input_rgb_folder_prefix)
train_file.preprocess_label(convert_label_to_int_and_subtract)

test_file.preprocess_name(add_input_rgb_folder_prefix)
test_file.preprocess_label(convert_label_to_int_and_subtract)

#flow
train_file_flow.preprocess_name(add_input_flow_folder_prefix)
train_file_flow.preprocess_label(convert_label_to_int_and_subtract)

test_file_flow.preprocess_name(add_input_flow_folder_prefix)
test_file_flow.preprocess_label(convert_label_to_int_and_subtract)


train_file.name = append_order_to_mica_split_file(train_file)
test_file.name = append_order_to_mica_split_file(test_file)

train_file_flow.name = append_order_to_mica_split_file(train_file_flow)
test_file_flow.name = append_order_to_mica_split_file(test_file_flow)

classifier_rgb = Classifier(
	train_file=train_file, 
	test_file=test_file, 
	class_ind=classInd, 
	classifier=config_c3d.clf,
	name=config_c3d.classifier_name,
	layer=config_c3d.layer,
	type_feature_file=config_c3d.type_feature_file
	)
classifier_flow = Classifier(
	train_file=train_file_flow, 
	test_file=test_file_flow, 
	class_ind=classInd, 
	classifier=config_c3d.clf,
	name=config_c3d.classifier_name,
	layer=config_c3d.layer,
	type_feature_file=config_c3d.type_feature_file
	)

# classify = ClassifyEarlyFusion(classifier_rgb, classifier_flow, concatenate_l2_norm) # early 
classify = VisualizeEarlyFusion(classifier_rgb, classifier_flow, concatenate_l2_norm) # early 
classify.execute()

# classify_flow = Classify(classifier_flow)
# classify_flow.execute()
