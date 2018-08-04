from __future__ import print_function
import sys
import os
import copy

import config_c3d
# from config.config_hmdb51 import *
from config.config_ucf101 import *
from Model.UCFSplitFile import *
from Model.ClassifierUsingProb import *
from Command.Classify import *
from Command.VisualizeFeature import *

train_ucf101 = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	# train_split_1_file_path, # hmdb51
	sample_train_path, #ucf101
	use_image=False)

test_file = UCFSplitFile(
	# r"(?P<name>.+) (?P<label>.+)", 
	r"(?P<label>.+)/(?P<name>\w+)",
	test_split_1_file_path,
	# sample_test_path,
	use_image=False)

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    class_ind_path)
classInd.load_name_and_label()

train_ucf101.load_name_and_label()
test_file.load_name_and_label()
print("Loaded: {} train label".format(len(train_ucf101.name)))
print("Loaded: {} test label".format(len(test_file.name)))

def add_input_folder_prefix(path):
	"""
	UCF101: rgb: <type feature>/<videoname>
			flow: <type feature>/<videoname>

	HMDB51: rgb: <type feature>/<classname>/<videoname>
			flow: <type feature>/<videoname>
	"""
	video_name = os.path.basename(os.path.splitext(path)[0])
	return os.path.join(rgb_feature_path, config_c3d.type_feature_file, video_name)
def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1
def convert_to_int(label):
	return int(label)
def convert_to_name(label):
	return classInd.convert_label_to_name(str(label))

train_ucf101.preprocess_name(add_input_folder_prefix)
train_ucf101.preprocess_label(convert_to_name) 

test_file.preprocess_name(add_input_folder_prefix) 
# test_file.preprocess_label(convert_to_name) # for hmdb51

classifier = Classifier(
	train_file=train_ucf101, 
	test_file=test_file, 
	class_ind=classInd, 
	classifier=config_c3d.clf,
	name=config_c3d.classifier_name,
	layer=config_c3d.layer,
	type_feature_file=config_c3d.type_feature_file
	)

plot_tsne = VisualizeFeature(classifier)
plot_tsne.execute()
