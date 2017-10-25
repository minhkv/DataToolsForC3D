from __future__ import print_function
import sys
import os
import config
sys.path.extend(["Model", "Command"])
from module_model import *
from module_command import *
from collections import defaultdict

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    os.path.join(config.asset_path, "classInd.txt"))
classInd.load_name_and_label()

with open('y_pred_ucf_finetune_split_3.csv', 'r') as pred, \
open('test_name_ucf_finetune_split_3.csv', 'r') as tn, \
open('y_true_ucf_finetune_split_3.csv', 'r') as tr:
    y_pred = [int(line.strip()) for line in pred.readlines()]
    y_true = [int(line.strip()) for line in tr.readlines()]
    test_name = [line.strip() for line in tn.readlines()]
"""Example: if test name i belong to class 0 (ApplyEyeMakeup) confused with class 1 (ApplyLipstick)
        then append to list of confused name of class 0 the record (name, label_pred)
y_true[i] = 0, y_pred[i] = 1, confused_class[class[0]].append((test_name[i], class[1]))
confused_name format:
    {
        class_name1: {
            false_class1: [video_name, ...],
            false_class2: [video_name, ...],
        }, 
        class_name2: {
            false_class1: [video_name, ...],
            false_class2: [video_name, ...],
        }, 
    }
"""

confused_name = {}
for class_name in classInd.name:
    confused_name[class_name] = defaultdict(list)

def preprocess_label(label):
    return classInd.convert_label_to_name(str(label + 1))
y_true = [preprocess_label(label) for label in y_true]
y_pred = [preprocess_label(label) for label in y_pred]

for name, true_label, predict_label in zip(test_name, y_true, y_pred):
    if true_label != predict_label:
        confused_name[true_label][predict_label].append(name)

with open('analyse_{}.txt'.format(config.classifier_name), 'w') as an:
    content = ""
    for class_name, false_classes in sorted(confused_name.items()):
        content += "{}\n".format(class_name)
        for false_class_name, false_video in false_classes.items():
            content += "\t{:20s}: {:5d}\n".format(false_class_name, len(false_video))


    an.write(content)
    