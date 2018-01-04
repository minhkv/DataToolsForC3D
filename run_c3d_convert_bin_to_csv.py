from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
import config
from Model.UCFSplitFile import *
from Command.CreateFeatureFolder import *
from Command.ConvertBinToCSV import *

all_ucf101 = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.train_split_file_path,
	use_image=False)
all_ucf101.load_name_and_label()
test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	config.test_split_file_path,
	use_image=False)
test_file.load_name_and_label()
all_ucf101.concatenate(test_file)
out_file = copy.copy(all_ucf101)
print("Loaded: {} label".format(len(all_ucf101.name)))

def add_input_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.output_feature_folder, "bin", os.path.basename(video_name))
def add_out_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(config.output_feature_folder, "csv", os.path.basename(video_name))
def subtract_label(label):
	return int(label) - 1
def dummy_label(label):
	return 0

all_ucf101.preprocess_name(add_input_folder_prefix)
all_ucf101.preprocess_label(dummy_label)
out_file.preprocess_name(add_out_folder_prefix)

create_output_folder = CreateFeatureFolder(out_file)
convert_bin_feature_to_csv = ConvertBinToCSV(all_ucf101, out_file, config.layer)

# create_output_folder.execute()
# convert_bin_feature_to_csv.execute()