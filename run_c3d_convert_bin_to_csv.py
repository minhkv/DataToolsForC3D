from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

c3d = C3D(
	root_folder="/home/minhkv/C3D/C3D-v1.0/", 
	c3d_mode=C3D_Mode.FEATURE_EXTRACTION,
	pre_trained=os.path.join(config.output_fine_tuned_net, "Split_2", "c3d_ucf101_finetune_whole_iter_20000"),
	use_image=False)
c3d.generate_prototxt()

all_ucf101 = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	"Asset/sample_train.txt",
	use_image=False)
all_ucf101.load_name_and_label()
test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	"Asset/sample_test.txt",
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
# def add_output_folder_prefix(path):
# 	folder_name = os.path.splitext(os.path.basename(path))[0]
# 	return os.path.join(config.output_folder, folder_name)

all_ucf101.preprocess_name(add_input_folder_prefix)
all_ucf101.preprocess_label(dummy_label)
out_file.preprocess_name(add_out_folder_prefix)

create_output_folder = CreateFeatureFolder(out_file)
create_output_folder.execute()

convert_bin_feature_to_csv = ConvertBinToCSV(all_ucf101, out_file, "fc6-1")
convert_bin_feature_to_csv.execute()