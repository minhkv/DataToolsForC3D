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

train_ucf101 = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
    # config.sample_train_file_path,
	config.train_split_1_file_path,
	use_image=False)

test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
    # config.sample_test_file_path,
	config.test_split_1_file_path,
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
	return os.path.join(config.output_feature_folder, "csv", os.path.basename(video_name))
def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1

train_ucf101.preprocess_name(add_input_folder_prefix)
train_ucf101.preprocess_label(subtract_label)
test_file.preprocess_name(add_input_folder_prefix)
test_file.preprocess_label(convert_and_subtract_label)

# def get_list_feature_in_folder(path, layer):
#     listfiles = glob.glob(os.path.join(path, "*" + layer + "*"))
#     return listfiles
# def get_all_feature_in_list_folders(list_folder, labels):
#     list_feature = []
#     list_label = []
#     for csv_folder, label in zip(list_folder, labels):
#         print ("[Info] Get all feature in: {}".format(os.path.basename(csv_folder)))
#         feature_in_folder = get_list_feature_in_folder(csv_folder, config.layer)
#         list_feature.extend(feature_in_folder)
#         list_label.extend([label] * len(feature_in_folder))
#     print("get {} features".format(len(list_feature)))
#     print("get {} labels".format(len(list_label)))
#     return list_feature, list_label

# train_ucf101.name, train_ucf101.label = get_all_feature_in_list_folders(train_ucf101.name, train_ucf101.label)
# test_file.name, test_file.label = get_all_feature_in_list_folders(test_file.name, test_file.label)

estimator = SVC(kernel="linear", C=0.025)
classifier = Classifier(estimator, train_ucf101, test_file, classInd, name="sport1m_split_1")

classify = Classify(classifier)
classify.execute()
