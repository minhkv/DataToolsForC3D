from __future__ import print_function
import config_c3d
import sys
from Model.C3D import *
from Model.UCFSplitFile import *

c3d_feature_extraction_ucf101 = C3D(
	root_folder=config_c3d.c3d_root, 
	c3d_mode=C3D_Mode.FEATURE_EXTRACTION_UCF101,
	pre_trained=config_c3d.pretrained,
	mean_file=config_c3d.mean_file,
	use_image=False)

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config_c3d.train_split_file_path,
	clip_size=16,
	chunk_list_syntax=config_c3d.input_chunk_list_line_syntax,
	chunk_list_file=config_c3d.input_chunk_file,
	use_image=False)

test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	config_c3d.test_split_file_path,
	clip_size=16,
	chunk_list_syntax=config_c3d.input_chunk_list_line_syntax,
	chunk_list_file=config_c3d.input_chunk_file,
	use_image=False)

out_file_empty = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config_c3d.empty_split_file_path,
	clip_size=16,
	chunk_list_syntax=config_c3d.output_chunk_list_line_syntax,
	chunk_list_file=config_c3d.output_chunk_file,
	use_image=False)