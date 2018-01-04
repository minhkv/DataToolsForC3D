from __future__ import print_function
import config
import sys
from Model.C3D import *
from Model.UCFSplitFile import *

c3d_feature_extraction_ucf101 = C3D(
	root_folder=config.c3d_root, 
	c3d_mode=C3D_Mode.FEATURE_EXTRACTION_UCF101,
	pre_trained=config.pretrained,
	mean_file=config.mean_file,
	use_image=False)

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.train_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=False)

test_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	config.test_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.input_chunk_list_line_syntax,
	chunk_list_file=config.input_chunk_file,
	use_image=False)

out_file_empty = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config.empty_split_file_path,
	clip_size=16,
	chunk_list_syntax=config.output_chunk_list_line_syntax,
	chunk_list_file=config.output_chunk_file,
	use_image=False)