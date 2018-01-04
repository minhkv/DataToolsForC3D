from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy

import config
import instance
from utils_c3d import *
from Model.C3D import *
from Model.MICASplitFile import *
from Command.CreateListPrefix import *
from Command.CreateFeatureFolder import *
from Command.FeatureExtraction import *
def dummy_label(label):
	return 0

c3d = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.FEATURE_EXTRACTION_UCF101,
	pre_trained=config.pretrained,
	mean_file=config.mean_file,
	model_config=config.model_config,
	# solver_config=config.solver_config,
	use_image=False)

all_file_mica = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_train_path,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=False)

test_file = MICASplitFile(
	config.mica_split_syntax, 
	path=config.mica_test_path,
	use_image=False)
classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    "/home/minhkv/datasets/Kinect_vis_annotate/ActionIndex.txt"
    )
out_file = MICASplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.empty_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.output_chunk_list_line_syntax,
	chunk_list_file=config.output_chunk_file,
	use_image=False)
classInd.load_name_and_label()
all_file_mica.load_name_and_label()
test_file.load_name_and_label()

print("Loaded: {} train label".format(len(all_file_mica.name)))
print("Loaded: {} test label".format(len(test_file.name)))
all_file_mica.concatenate(test_file)
out_file.concatenate(all_file_mica)

all_file_mica.preprocess_name(add_input_folder_prefix)
all_file_mica.preprocess_label(dummy_label)

out_file.preprocess_name(add_output_folder_prefix)
out_file.name = append_order_to_mica_split_file(out_file)

createList = CreateListPrefix(
	split_file=all_file_mica,
	output_feature_file=out_file, 
	)

create_output_folder = CreateFeatureFolder(out_file)
feature_extract = FeatureExtraction(c3d)

# Execute
# c3d.generate_prototxt()
# createList.execute()
# create_output_folder.execute()
# feature_extract.execute()
