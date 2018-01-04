from __future__ import print_function
from Model.UCFSplitFile import *
import os, re

"""
    Annotation HMDB51 structure:
            <class_name>_test_split<1-3>.txt
        Line syntax:
            <video_name> <0|1|2>
                0: not include in training/testing
                1: for training
                2: for testing   
            70 for training, 30 for testing each txt file     
        Video folder structure:
            data/
                <class_name_1>/
                    <video_1>.avi
                    ...
                ...
    
    Output:
        - Create class_ind.txt, train0<1-3>.txt, test0<1-3>.txt  
        - Line syntax:
            <class_name>/<video_name> <label>
"""
hmdb51_annotation = "/home/minhkv/datasets_old/hmdb51/testTrainMulti_7030_splits"
hmdb51_video_path = "/home/minhkv/datasets_old/hmdb51/data"
hmdb51_split_path = "/home/minhkv/script/Run_C3D/DataToolsForC3D/Asset/hmdb51_split"
annotation_name_syntax = r"(?P<class_name>.+)_test_split(?P<num_split>\w+).txt"
annotation_line_syntax = r"(?P<video_name>.+) (?P<id>.+)"
train01_path = os.path.join(hmdb51_split_path, "train01.txt") 
train02_path = os.path.join(hmdb51_split_path, "train02.txt")
train03_path = os.path.join(hmdb51_split_path, "train03.txt")

test01_path = os.path.join(hmdb51_split_path, "test01.txt")
test02_path = os.path.join(hmdb51_split_path, "test02.txt")
test03_path = os.path.join(hmdb51_split_path, "test03.txt")
train_files = [train01_path, train02_path, train03_path]
test_files = [test01_path, test02_path, test03_path]
class_ind = os.path.join(hmdb51_split_path, "class_ind.txt")

list_action = sorted(os.listdir(hmdb51_video_path))
list_annotation = sorted(os.listdir(hmdb51_annotation))
map_name_id = {}
with open(class_ind, "w") as fp:
    for i, action in enumerate(list_action):
        fp.write("{} {}\n".format(i, action))
        map_name_id[action] = i
m = re.compile(annotation_name_syntax)
m_line = re.compile(annotation_line_syntax)
index = 0

train = []
test = []
count = 0
for annotation in list_annotation:
    d = m.search(annotation).groupdict()
    if int(d["num_split"]) != index + 1:
        continue
    class_name = d["class_name"]
    
    with open(os.path.join(hmdb51_annotation, annotation), "r") as fp:
        content = fp.readlines()
        for line in content:
            d_line = m_line.search(line).groupdict()
            print(d_line)
            line_id = int(d_line["id"])
            count += 1
            item = "{}/{} {}".format(class_name, d_line["video_name"], map_name_id[d["class_name"]])
            if line_id == 1:
                train.append(item)
            elif line_id == 2:
                test.append(item)

with open(train_files[index], "w") as tr, \
    open(test_files[index], "w") as te:
    tr.write("\n".join(train))
    te.write("\n".join(test))
print(count)
print(len(glob.glob(os.path.join(hmdb51_video_path, "*/*.avi"))))
