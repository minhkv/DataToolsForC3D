from __future__ import print_function
import sys
import os
import copy
import numpy as np
import config
import instance
from Model.MICASplitFile import *
from Model.UCFSplitFile import *

def convert_label_to_int(label):
    label = label.replace('\xef\xbb\xbf', '')
    return int(label)

def add_annotation_prefix(path):
	name = os.path.join(config.mica_annotation_path, path.strip() + ".txt")
	return name


classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    "/home/minhkv/datasets/Kinect_vis_annotate/ActionIndex.txt"
    )

file_mapping = UCFSplitFile(
    r"(?P<label>.+) (?P<name>.+)", 
    "/home/minhkv/datasets/Kinect_vis_annotate/file_mapping.txt"
)

classInd.load_name_and_label()
file_mapping.load_name_and_label()
file_mapping.preprocess_name(add_annotation_prefix)
file_mapping.preprocess_label(convert_label_to_int)
train_annotation = []
test_annotation = []

for annotation, index in zip(file_mapping.name, file_mapping.label):
    if index % 2 == 0:
        test_annotation.append(annotation)
    else:
        train_annotation.append(annotation)

def write_split_to_file_from_annotation(list_annotation, file):
    with open(file, 'w+') as fp:
        for annotation in list_annotation:
            split_file = MICASplitFile(
                r"(?P<label>.+);(?P<start_in_video>.+);(?P<end_in_video>\w+)", 
                annotation,
                use_image=False)
            
            split_file.load_name_and_label()
            split_file.preprocess_label(convert_label_to_int)
            lines = []
            i = 1
            for label, start_in_video, end_in_video, ann in zip(split_file.label, split_file.start_in_video, split_file.end_in_video, split_file.name):
                lines.append("{} {} {} {} {}\n".format(label, start_in_video, end_in_video, os.path.basename(ann), i))
                i += 1
            fp.write(''.join(lines))
        
write_split_to_file_from_annotation(train_annotation, os.path.join(config_c3d.asset_path, 'mica_train.txt'))
write_split_to_file_from_annotation(test_annotation, os.path.join(config_c3d.asset_path, 'mica_test.txt'))
