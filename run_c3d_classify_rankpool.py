from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy

sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

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

def add_input_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.output_feature_folder, os.path.basename(video_name))
def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1

train_ucf101.preprocess_name(add_input_folder_prefix)
train_ucf101.preprocess_label(subtract_label)
test_file.preprocess_name(add_input_folder_prefix)
test_file.preprocess_label(convert_and_subtract_label)

classifier = ClassifierForRankpool(
	train_file=train_ucf101, 
	test_file=test_file, 
	class_ind=classInd, 
	classifier=config.clf,
	name=config.classifier_name,
	layer=config.layer,
	type_feature_file=config.type_feature_file
	)

classify = Classify(classifier)
classify.execute()