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
	pre_trained="/home/minhkv/pre-trained/conv3d_deepnetA_sport1m_iter_1900000",
	use_image=False)
c3d.generate_prototxt()

train_file = UCFSplitFile(
	r"(?P<name>.+) (?P<label>\w+)", 
	"Asset/trainlist02.txt",
	use_image=False)


train_file.load_name_and_label()
print("Loaded: {} label".format(len(train_file.name)))

def add_input_folder_prefix(path):
	return os.path.join(config.ucf101_video_folder, os.path.basename(path))
def subtract_label(label):
	return int(label) - 1
# def add_output_folder_prefix(path):
# 	folder_name = os.path.splitext(os.path.basename(path))[0]
# 	return os.path.join(config.output_folder, folder_name)

train_file.preprocess_name(add_input_folder_prefix)
train_file.preprocess_label(subtract_label)

print("Counting frame")
train_file.count_frame()

createList = CreateListPrefix(
	split_file=train_file,
	output_feature_file=train_file, 
	use_image=False
	)
createList.execute()

# compute_volume_mean = ComputeVolumeMean(c3d)
# compute_volume_mean.execute()

finetune = Finetune(c3d)
finetune.execute()