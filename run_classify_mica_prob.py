from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
sys.path.extend(["Model", "Command"])
import config
from Model.UCFSplitFile import *
from Model.MICASplitFile import *
from Model.ClassifierUsingProb import *
from Command.Classify import *
import instance

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

train_file = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_train_path,
	use_image=False)

test_file = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_test_path,
	use_image=False)
classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    "/home/minhkv/datasets/Kinect_vis_annotate/ActionIndex.txt"
    )

classInd.load_name_and_label()
train_file.load_name_and_label()
test_file.load_name_and_label()

print("Loaded: {} train label".format(len(train_file.name)))
print("Loaded: {} test label".format(len(test_file.name)))

train_file.preprocess_name(add_output_folder_prefix)
train_file.preprocess_label(convert_label_to_int_and_subtract)

test_file.preprocess_name(add_output_folder_prefix)
test_file.preprocess_label(convert_label_to_int_and_subtract)

train_file.name = append_order_to_mica_split_file(train_file)
test_file.name = append_order_to_mica_split_file(test_file)
check_empty(train_file.name)
check_empty(test_file.name)
# print('\n'.join(train_file.name))

classifier = ClassifierUsingProb(
	train_file=train_file, 
	test_file=test_file, 
	class_ind=classInd, 
	classifier=config.clf,
	name=config.classifier_name,
	layer=config.layer,
	type_feature_file=config.type_feature_file
	)

classify = Classify(classifier)

# classify.execute()
