from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy

import config_c3d
from Model.UCFSplitFile import *
from Model.ClassifierUsingProb import *
from Command.Classify import *

train_ucf101 = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config_c3d.train_split_file_path,
	use_image=False)

test_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>.+)", 
	config_c3d.test_split_file_path,
	use_image=False)

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    config_c3d.class_ind_path)
classInd.load_name_and_label()

train_ucf101.load_name_and_label()
test_file.load_name_and_label()
print("Loaded: {} train label".format(len(train_ucf101.name)))
print("Loaded: {} test label".format(len(test_file.name)))

def add_input_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	# return os.path.join(config_c3d.output_feature_folder, config_c3d.type_feature_file, os.path.basename(video_name))
	return os.path.join(config_c3d.output_feature_folder, config_c3d.type_feature_file, path)
def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1
def convert_to_int(label):
	return int(label)

train_ucf101.preprocess_name(add_input_folder_prefix)
train_ucf101.preprocess_label(convert_to_int)
test_file.preprocess_name(add_input_folder_prefix)
test_file.preprocess_label(convert_to_int)

classifier = ClassifierUsingProb(
	train_file=train_ucf101, 
	test_file=test_file, 
	class_ind=classInd, 
	classifier=config_c3d.clf,
	name=config_c3d.classifier_name,
	layer=config_c3d.layer,
	type_feature_file=config_c3d.type_feature_file
	)

classify = Classify(classifier)
classify.execute()
