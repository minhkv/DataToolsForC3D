from __future__ import print_function
import sys
import os
import config
sys.path.extend(["Model", "Command"])
from module_model import *
from module_command import *
from collections import defaultdict
import operator
from sklearn.metrics import classification_report
from utils_c3d import convert_leaf_to_group

report_folder = "{0}/report_{1}".format(config.report_folder, config.classifier_name)
y_pred_file = os.path.join(report_folder, 'y_pred_{}.csv'.format(config.classifier_name))
y_true_file = os.path.join(report_folder, 'y_true_{}.csv'.format(config.classifier_name))
test_name_file = os.path.join(report_folder, 'test_name_{}.csv'.format(config.classifier_name))
output_report = 'analyse_group_{}.txt'.format(config.classifier_name)
output_file = os.path.join(report_folder, output_report)


# classInd = UCFSplitFile(
#     r"(?P<label>.+) (?P<name>\w+)", 
#     config.classInd_file_path)
classInd = MICASplitFile(
	r"(?P<label>.+) (?P<name>.+)",
	path='/home/minhkv/datasets/Kinect_vis_annotate/ClassTree.txt'
)
classInd.load_name_and_label()
with open(y_pred_file, 'r') as pred, \
open(test_name_file, 'r') as tn, \
open(y_true_file, 'r') as tr:
    y_pred = [int(line.strip()) for line in pred.readlines()]
    y_true = [int(line.strip()) for line in tr.readlines()]
    test_name = [line.strip() for line in tn.readlines()]

y_pred = [convert_leaf_to_group(y) for y in y_pred]
y_true = [convert_leaf_to_group(y) for y in y_true]

y_pred = [classInd.convert_label_to_name(str(y + 1)) for y in y_pred]
y_true = [classInd.convert_label_to_name(str(y + 1)) for y in y_true]
with open(output_file, 'w') as an:
    an.write(classification_report(y_true, y_pred, digits=7))