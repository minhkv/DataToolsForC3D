from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
import numpy as np
import config
from Model.UCFSplitFile import *
from Model.Classifier import *
from Command.Classify import *

def add_input_rgb_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.feature_folder_sport1m, config.type_feature_file, os.path.basename(video_name))
def add_input_flow_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.feature_folder_flow, config.type_feature_file, os.path.basename(video_name))

def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1

def fuse_function(vect1, vect2):
	return np.array(vect1) + np.array(vect2)

train_ucf101 = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.train_split_file_path,
	use_image=False)

test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	config.test_split_file_path,
	use_image=False)

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    os.path.join(config.asset_path, "classInd.txt"))
classInd.load_name_and_label()

train_ucf101.load_name_and_label()
test_file.load_name_and_label()
print("Loaded: {} train label".format(len(train_ucf101.name)))
print("Loaded: {} test label".format(len(test_file.name)))

train_ucf101_flow = copy.copy(train_ucf101)
test_ucf101_flow = copy.copy(test_file)

#rgb
train_ucf101.preprocess_name(add_input_rgb_folder_prefix)
train_ucf101.preprocess_label(subtract_label)

test_file.preprocess_name(add_input_rgb_folder_prefix)
test_file.preprocess_label(convert_and_subtract_label)

#flow
train_ucf101_flow.preprocess_name(add_input_flow_folder_prefix)
train_ucf101_flow.preprocess_label(subtract_label)

test_ucf101_flow.preprocess_name(add_input_flow_folder_prefix)
test_ucf101_flow.preprocess_label(convert_and_subtract_label)

classifier_rgb = Classifier(
	train_file=train_ucf101, 
	test_file=test_file, 
	class_ind=classInd, 
	classifier=config.clf,
	name=config.classifier_name,
	layer=config.layer,
	type_feature_file=config.type_feature_file
	)
classifier_flow = Classifier(
	train_file=train_ucf101_flow, 
	test_file=test_ucf101_flow, 
	class_ind=classInd, 
	classifier=config.clf,
	name=config.classifier_name,
	layer=config.layer,
	type_feature_file=config.type_feature_file
	)

classify = Classify(classifier_rgb, classifier_flow, fuse_function)
# classify.execute()
