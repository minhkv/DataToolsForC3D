from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
from sklearn.svm import SVC
sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

bin_train = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
    # config.sample_train_file_path,
	config.train_split_3_file_path,
	use_image=False)

bin_test = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
    # config.sample_test_file_path,
	config.test_split_3_file_path,
	use_image=False)

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    os.path.join(config.asset_path, "classInd.txt"))
classInd.load_name_and_label()

bin_train.load_name_and_label()
bin_test.load_name_and_label()


def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1

def add_input_folder_prefix_bin(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.output_feature_folder, "bin", os.path.basename(video_name))

bin_train.preprocess_name(add_input_folder_prefix_bin)
bin_train.preprocess_label(subtract_label)
bin_test.preprocess_name(add_input_folder_prefix_bin)
bin_test.preprocess_label(convert_and_subtract_label)


estimator = SVC(kernel="linear", C=0.025)
classifier_bin = ClassifierUsingBin(estimator, bin_train, bin_test, classInd, name="ucf_finetune_split_3_using_bin")

classifier_bin.load_train_test_split()
classify = Classify(classifier_bin)
classify.execute()
