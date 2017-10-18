from __future__ import print_function
from RemoteControl import *
import sys
import os
sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

c3d = C3D(
	root_folder="/home/minhkv/C3D/C3D-v1.0/", 
	input_prefix=os.path.join(config.temp, "input.txt"), 
	pre_trained=os.path.join("c3d_ucf101_finetune_whole_iter_20000"),
	training=False,
	use_image=False)
c3d.generate_prototxt()

test_file = UCFSplitFile(
	r"(?P<label>.+)/(?P<name>.+)", 
	os.path.join(config.asset_path, "sample_test.txt"),
	use_image=False)
test_file.load_name_and_label()
print("Loaded: {} label".format(len(test_file.name)))

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    os.path.join(config.asset_path, "classInd.txt"))
classInd.load_name_and_label()

def add_input_folder_prefix(path):
	return os.path.join(config.ucf101_video_folder, os.path.basename(path))
def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1
# def add_output_folder_prefix(path):
# 	folder_name = os.path.splitext(os.path.basename(path))[0]
# 	return os.path.join(config.output_folder, folder_name)

test_file.preprocess_name(add_input_folder_prefix)
test_file.preprocess_label(convert_and_subtract_label)
print("Counting frame")
test_file.count_frame()

createList = CreateListPrefix(
	split_file=test_file,
	output_feature_file=None, 
	use_image=False
	)
createList.execute()

# compute_volume_mean = ComputeVolumeMean(c3d)
# compute_volume_mean.execute()

# finetune = Finetune(c3d)
# finetune.execute()

test_net = TestNet(c3d)
code = test_net.execute()
print(code)