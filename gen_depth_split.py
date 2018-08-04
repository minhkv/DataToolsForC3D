from __future__ import print_function
from Model.UCFSplitFile import *
from config.config_mica import *
import re, os
mica_annotation_path = "/home/minhkv/datasets/Kinect_clips/gtv2"
"""
        order start from 0
        output order start from 1
"""

def find_start_end(annotation_name, order):
    with open(os.path.join(mica_annotation_path, mica_annotation_path, annotation_name + ".txt")) as an:
        content = an.readlines()
        return (content[order].strip().split(';'))


depth_train_path = "/home/minhkv/datasets/Kinect_clips/test_depth_lst_v2.txt"
syntax = r"(?P<path>.+)/(?P<annotation>.+)/(?P<order>.+) (?P<num_frame>.+) (.+)"
mica_depth = "/home/minhkv/script/MICA_C3D/DataToolsForC3D/Asset/mica_split/mica_depth_test.txt"

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    "/home/minhkv/datasets/Kinect_vis_annotate/ActionIndex.txt"
)
classInd.load_name_and_label()

new_content = []
with open(depth_train_path, "r") as dtp:
    
    content = dtp.readlines()
    for line in content:
        m = re.compile(syntax)
        d = m.search(line).groupdict()
        ann_line = find_start_end(d['annotation'], int(d['order']))
        label = ann_line[0]
        start_in_video = ann_line[1]
        end_in_video = ann_line[2]
        new_content.append("{} {} {} {} {}".format(label, start_in_video, end_in_video, d['annotation'] + ".txt", int(d['order']) + 1))

with open(mica_depth, "w") as fp:
    fp.write('\n'.join(new_content))
        

# ann = "20171221NguyenThiThao_21-12-2017__15-00-47"
# print(find_start_end(ann, 0))