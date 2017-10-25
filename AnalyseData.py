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

with open('y_pred_{}.csv'.format(config.classifier_name), 'r') as pred, \
open('test_name_{}.csv'.format(config.classifier_name), 'r') as tn, \
open('y_true_{}.csv'.format(config.classifier_name), 'r') as tr:
    y_pred = [int(line.strip()) for line in pred.readlines()]
    y_true = [int(line.strip()) for line in tr.readlines()]
    test_name = [line.strip() for line in tn.readlines()]
"""Example: if test name i belong to class 0 (ApplyEyeMakeup) confused with class 1 (ApplyLipstick)
        then append to list of confused name of class 0 the record (name, label_pred)
y_true[i] = 0, y_pred[i] = 1, confused_class[class[0]].append((test_name[i], class[1]))
confused_name format:
    {
        class_name1: {
            num_video: 33,
            false_class1: [video_name, ...],
            false_class2: [video_name, ...],
        }, 
        class_name2: {
            num_video: 34,
            false_class1: [video_name, ...],
            false_class2: [video_name, ...],
        }, 
    }
"""

confused_name = {}
for class_name in classInd.name:
    confused_name[class_name] = defaultdict(list)
    confused_name[class_name]['num_video'] = 0

def preprocess_label(label):
    return classInd.convert_label_to_name(str(label + 1))
y_true = [preprocess_label(label) for label in y_true]
y_pred = [preprocess_label(label) for label in y_pred]

for name, true_label, predict_label in zip(test_name, y_true, y_pred):
    confused_name[true_label]['num_video'] += 1
    if true_label != predict_label:
        confused_name[true_label][predict_label].append(name)

with open('analyse_{}.txt'.format(config.classifier_name), 'w') as an:
    content = ""
    for class_name, info in sorted(confused_name.items()):
        content += "{:20s} {:5d}\n".format(class_name, info['num_video'])
        for false_class_name, false_video in info.items():
            if isinstance(false_video, int): 
                continue
            content += "\t{:20s}: {:5d}\n".format(false_class_name, len(false_video))
    an.write(content)
    