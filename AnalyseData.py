from __future__ import print_function
import sys
import os
import config
sys.path.extend(["Model", "Command"])
from module_model import *
from module_command import *

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    os.path.join(config.asset_path, "classInd.txt"))
classInd.load_name_and_label()

with open('y_pred_ucf_finetune_split_3.csv', 'r') as pred, \
open('y_true_ucf_finetune_split_3.csv', 'r') as tr:
    y_pred = [int(line.strip()) for line in pred.readlines()]
    y_true = [int(line.strip()) for line in tr.readlines()]

# Example: if test name i belong to class 0 (ApplyEyeMakeup) confused with class 1 (ApplyLipstick)
#         then append to list of confused name of class 0 the record (name, label_pred)
# y_true[i] = 0, y_pred[i] = 1, confused_class[class[0]].append((test_name[i], class[1]))

confused_name = {}
for class_name in classInd.name:
    confused_name[class_name] = []

# for test_label, test_predict in zip(test_file.name, y_true, y_pred):
