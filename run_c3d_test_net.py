from __future__ import print_function
from RemoteControl import *
import sys
import os
import config

from Model.C3D import *
from Model.UCFSplitFile import *
from Command.CreateListPrefix import *
from Command.TestNet import *

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.TEST_FINE_TUNED_NET,
	model_config=config.model_config,
	pre_trained=config.pretrained,
	mean_file=config.mean_file,
	use_image=True)


test_file = UCFSplitFile(
	config.test_file_line_syntax, 
	config.test_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=True,
	type_image="png")
test_file.load_name_and_label()
print("Loaded: {} label".format(len(test_file.name)))

classInd = UCFSplitFile(
    config.classInd_file_line_syntax, 
    config.classInd_file_path)
classInd.load_name_and_label()

def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	return os.path.join(config.ucf101_stack_tvl1_folder, name)
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
	output_feature_file=None
	)
test_net = TestNet(c3d)

# c3d.generate_prototxt()
# createList.execute()
# test_net.execute()