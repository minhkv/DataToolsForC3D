from __future__ import print_function
import os, sys, numpy as np
from sklearn.metrics import accuracy_score, classification_report
from collections import OrderedDict
from config.config_mica import *
from Model.MICASplitFile import *
from utils_c3d import *

def read_prob_file(file_name, splitter=' ', line_from_zero=False):
    print(file_name)
    prob = []
    label = []
    name = []
    pred = []
    num_line = []
    with open(file_name, "r") as f_test:
        content = f_test.readlines()
        for line in content:
            if("annotation" in line):
                continue
            item = line.replace('\xef\xbb\xbf', '').strip().split(splitter)
            name.append(item[0])
            if(line_from_zero):
                num_line.append(int(item[1]) + 1)
            else:
                num_line.append(int(item[1]))
            label.append(int(item[2]) - 1)
            prob.append([float(i) for i in item[3:]])
        prob = np.array(prob)
    pred = [np.argmax(p) for p in prob]
    return name, num_line, label, prob

def create_dict_result(name, num_line, prob, line_from_zero=False):
    mapping = OrderedDict()
    for n, l in zip(test_file.name, test_file.segment_order):
        mapping["{}__{}".format(n, l)] = np.zeros(prob.shape[1])
    print(len(mapping.items()))
    for n, l, p in zip(name, num_line, prob):
        if line_from_zero:
            mapping["{}__{}".format(n, l + 1)] = p
        else:
            mapping["{}__{}".format(n, l)] = p
    print(len(mapping.items()))
    return mapping

test_file = MICASplitFile(
        mica_split_syntax, 
        path=mica_test_path,
        use_image=False)
test_file.load_name_and_label()

def split_ext(path):
    return path.strip().split('.')[0]
def convert_label_to_int(label):
	label = label.replace('\xef\xbb\xbf', '')
	return int(label) - 1
test_file.preprocess_name(split_ext)
test_file.preprocess_label(convert_label_to_int)


# file_depth = "report/report_classify_mica_depth_v5_10000_20classes/prob_test_classify_mica_depth_v5_10000_20classes.csv"
# file_rgb = "report/report_classify_mica_fc6_20classes/prob_test_classify_mica_fc6_20classes.csv"
# file_cnn = "report/last_fusion_20.csv"

# file_depth = "report/report_classify_mica_depth_v5_10000_group/prob_test_classify_mica_depth_v5_10000_group.csv"
# file_rgb = "report/report_classify_mica_fc6_group/prob_test_classify_mica_fc6_group.csv"
# file_cnn = "report/last_fusion_6.csv"
# test_file.preprocess_label(convert_leaf_to_group)

file_depth = "report/report_classify_mica_depth_v5_10000_root/prob_test_classify_mica_depth_v5_10000_root.csv"
file_rgb = "report/report_classify_mica_fc6_root/prob_test_classify_mica_fc6_root.csv"
file_cnn = "report/last_fusion_2.csv"
test_file.preprocess_label(convert_leaf_to_root)

name_rgb, num_line_rgb, label_rgb, prob_rgb = read_prob_file(file_rgb)
name_depth, num_line_depth, label_depth, prob_depth = read_prob_file(file_depth, line_from_zero=True)
name_cnn, num_line_cnn, label_cnn, prob_cnn = read_prob_file(file_cnn, splitter=',')

result_rgb = create_dict_result(name_rgb, num_line_rgb, prob_rgb)
result_depth = create_dict_result(name_depth, num_line_depth, prob_depth)
result_cnn = create_dict_result(name_cnn, num_line_cnn, prob_cnn)

# s1 = set(num_line_rgb)
# s2 = set(num_line_depth)
# print(s1)
# print(s2)

# result = result_depth
# pred = []
# for key, value in result.items():
#     if result_rgb.has_key(key):
#         result[key] = [i + j for i, j in zip(result[key], result_rgb[key])]
#     # print(np.array(result[key]).shape)
#     pred.append(np.argmax(result[key]))

pred = [np.argmax(value) for key, value in result_rgb.items()]
print(classification_report(test_file.label, pred, digits=7))
print(accuracy_score(test_file.label, pred))

pred = [np.argmax(value) for key, value in result_depth.items()]
print(classification_report(test_file.label, pred, digits=7))
print(accuracy_score(test_file.label, pred))

pred = [np.argmax(value) for key, value in result_cnn.items()]
print(classification_report(test_file.label, pred, digits=7))
print(accuracy_score(test_file.label, pred))
