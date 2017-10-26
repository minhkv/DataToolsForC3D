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
	c3d_mode=C3D_Mode.TEST_FINE_TUNED_NET,
	pre_trained=os.path.join("c3d_ucf101_finetune_whole_iter_2000"),
	mean_file=config.mean_file,
	use_image=False)
c3d.generate_prototxt()

test_file = UCFSplitFile(
	config.test_file_line_syntax, 
	config.test_split_file_path,
	use_image=False)
test_file.load_name_and_label()
print("Loaded: {} label".format(len(test_file.name)))

classInd = UCFSplitFile(
    config.classInd_file_line_syntax, 
    config.classInd_file_path)
classInd.load_name_and_label()

def add_input_folder_prefix(path):
	return os.path.join(config.ucf101_video_folder, os.path.basename(path))
def subtract_label(label):
	return int(label) - 1
def convert_and_subtract_label(label):
	return int(classInd.convert_name_to_label(label)) - 1

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

test_net = TestNet(c3d)
code = test_net.execute()
print(code)