from __future__ import print_function
from RemoteControl import *
import sys
import os
import config_c3d

from Model.C3D import *
from Model.UCFSplitFile import *
from Command.CreateListPrefix import *

c3d = C3D(
	root_folder=config_c3d.c3d_root, 
	c3d_mode=C3D_Mode.TRAINING,
	pre_trained=config_c3d.pretrained,
	mean_file=config_c3d.mean_file,
	use_image=config_c3d.use_image)
c3d.generate_prototxt()

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	config_c3d.train_split_file_path,
	use_image=config_c3d.use_image)


train_file.load_name_and_label()
print("Loaded: {} label".format(len(train_file.name)))

def add_input_folder_prefix(path):
	return os.path.join(config_c3d.input_folder_prefix, os.path.basename(path))
def subtract_label(label):
	return int(label) - 1

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(subtract_label)

print("Counting frame")
train_file.count_frame()

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=None, 
	)
# createList.execute()

# finetune = Finetune(c3d)
# finetune.execute()