from __future__ import print_function
import sys
import os
import copy
import config_c3d
from config.config_mica import *
import instance
from Model.C3D import *
from Model.UCFSplitFile import *
from Model.MICASplitFile import *
from Command.CreateListPrefix import *
from Command.TestNet import *
from Command.CheckBadImage import *

def add_input_folder_prefix(path):
	name = os.path.basename(path).split('.')[0]
	name = os.path.join(stack_flow_mica, name)
	return name
def convert_label_to_int_and_subtract(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1
def append_order_to_mica_split_file(split_file):	
	new_name = []
	for name, order in zip(split_file.name, split_file.segment_order):
		new_name.append(os.path.join(name, str(int(order))))
	return new_name

c3d = C3D(
	root_folder="/home/minhkv/script/MICA_C3D/C3D_FLOW/C3D-v1.0", 
	c3d_mode=C3D_Mode.TEST_FINE_TUNED_NET,
	pre_trained="/home/minhkv/pre-trained/mica/merged_flow/conv3d_mica_flow_train_iter_10000",
	mean_file=mean_file_flow,
	model_config=model_flow_test,
	use_image=True)

test_file = MICASplitFile(
	mica_split_syntax, 
	# path="/home/minhkv/script/MICA_C3D/DataToolsForC3D/Asset/mica_split/mica_depth_test.txt",
	path=mica_test_path,
	chunk_list_syntax=config_c3d.input_chunk_list_line_syntax,
	chunk_list_file=config_c3d.input_chunk_file,
	use_image=True)

test_file.load_name_and_label()
print("Loaded: {} label".format(len(test_file.name)))

test_file.preprocess_name(add_input_folder_prefix)
test_file.name = append_order_to_mica_split_file(test_file)
test_file.preprocess_label(convert_label_to_int_and_subtract)
test_file.count_frame()

createList = CreateListPrefix(
	split_file=test_file,
	output_feature_file=None, 
	)
test_net = TestNet(c3d)

# checkImage = CheckBadImage(test_file)
# checkImage.execute()

c3d.generate_prototxt()
createList.execute()
test_net.execute()

