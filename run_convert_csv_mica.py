from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
import config
from Model.MICASplitFile import *
from Command.CreateFeatureFolder import *
from Command.ConvertBinToCSV import *
import instance
from utils_c3d import *


def add_input_folder_annotation(path):
	name = os.path.join(config.mica_annotation_path, path + ".txt")
	return name

def add_output_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.output_feature_folder, "bin", name)
	return name
def add_output_csv_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(config.output_feature_folder, "csv", name)
	return name
def convert_label_to_int(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label)

all_file_mica = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_train_path,
	)
test_file = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_test_path)

out_file = MICASplitFile(
	config.mica_split_syntax, 
	path="")

all_file_mica.load_name_and_label()
test_file.load_name_and_label()
all_file_mica.concatenate(test_file)
out_file.concatenate(all_file_mica)
print("Loaded: {} label".format(len(all_file_mica.name)))

all_file_mica.preprocess_name(add_output_folder_prefix)
all_file_mica.name = append_order_to_mica_split_file(all_file_mica)

out_file.preprocess_name(add_output_csv_folder_prefix)
out_file.name = append_order_to_mica_split_file(out_file)

create_output_folder = CreateFeatureFolder(out_file)
convert_bin_feature_to_csv = ConvertBinToCSV(all_file_mica, out_file, config.layer)
# Execute
# create_output_folder.execute()
# convert_bin_feature_to_csv.execute()
