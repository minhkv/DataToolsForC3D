from __future__ import print_function
from RemoteControl import *
import sys
import os
import copy
import shutil

sys.path.extend(["Model", "Command"])
import config
from module_model import *
from module_command import *

fc6_folder = '/home/minhkv/datasets/feature/minhkv/sport1m/bin'
rankpool_folder = '/home/minhkv/feature/sport1m_rankpooling_w'
dest_fc6 = '/home/minhkv/feature/analyse_w/fc6-1'
dest_rankpool = '/home/minhkv/feature/analyse_w/rankpool'

list_w_file = 'Asset/analyse_w_list.txt'

w_file = UCFSplitFile(
	r"(?P<label>\w+)/(?P<name>.+)", 
	list_w_file,
	use_image=False)

w_file.load_name_and_label()
print("Loaded: {} label".format(len(w_file.name)))
fc6_file = copy.copy(w_file)

def add_input_folder_prefix(path):
	video_name = os.path.splitext(path)[0]
	return os.path.join(input_folder, os.path.basename(video_name))
def add_output_folder_prefix(path):
    video_name = os.path.splitext(path)[0]
    return os.path.join(output_folder, os.path.basename(video_name))

def copy_list_file(src, dest, list_file):
    input_folder = src
    output_folder = dest
    print(input_folder)
    src_file = copy.copy(list_file)
    dest_file = copy.copy(list_file)
    src_file.preprocess_name(add_input_folder_prefix)
    dest_file.preprocess_name(add_output_folder_prefix)
    for src_feat, dest_feat in zip(src_file.name, dest_file.name):
        print(src_feat)
        if os.path.isdir(src_feat):
            if not os.path.isdir(dest_feat):
                os.mkdir(dest_feat)
            for feat in os.listdir(src_feat):
                shutil.copy(os.path.join(src_feat, feat), dest_feat)
        else:
            shutil.copy(src_feat + ".csv", dest_feat + ".csv")
    return
def multiply_w(w_folder, feature_folder, w_file):
    pass
input_folder = fc6_folder
output_folder = dest_fc6
copy_list_file(fc6_folder, dest_fc6, w_file)
input_folder = rankpool_folder
output_folder = dest_rankpool
copy_list_file(rankpool_folder, dest_rankpool, w_file)
