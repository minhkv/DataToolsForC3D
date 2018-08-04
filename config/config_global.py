import os
import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.kernel_approximation import AdditiveChi2Sampler, RBFSampler, SkewedChi2Sampler
from kernel.pairwise import *
import numpy as np
from sklearn import pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer

asset_path = os.path.abspath("Asset")
temp = os.path.abspath("Asset/tmp")
output_fine_tuned_net = os.path.abspath("Finetuned_net")
# c3d_root = "/home/minhkv/C3D/C3D-v1.0/"
# c3d_root = "/home/minhkv/script/Run_C3D/C3D/C3D-v1.0" # for png image
c3d_mica_flow = "/home/minhkv/script/MICA_C3D/C3D_FLOW/C3D-v1.0" # for mica flow "color.avi_05d.jpg"

input_chunk_list_line_syntax = "{0} {1} {2}\n"
output_chunk_list_line_syntax = "{0}/{1:06d}\n"

train_file_line_syntax = r"(?P<name>.+) (?P<label>\w+)"
test_file_line_syntax = r"(?P<label>.+)/(?P<name>.+)"
classInd_file_line_syntax = r"(?P<label>.+) (?P<name>\w+)"

input_chunk_file = os.path.join(temp, "input.txt")
output_chunk_file = os.path.join(temp, "output.txt")

empty_split_file_path = os.path.join(asset_path, "empty.txt")
sample_train_file_path = os.path.join(asset_path, "sample_train.txt")
sample_test_file_path = os.path.join(asset_path, "sample_test.txt")

lost_train_file_path = os.path.join(asset_path, "lost_train.txt")
lost_test_file_path = os.path.join(asset_path, "lost_test.txt")

mean_mica_split = os.path.join(temp, "mean_mica_split.binaryproto")
report_folder = os.path.abspath("report")

model_feature_extractor_sport1m = os.path.join(asset_path, "c3d_sport1m_feature_extractor.prototxt")
pre_trained_sport1m="/home/minhkv/pre-trained/conv3d_deepnetA_sport1m_iter_1900000"
feature_folder_sport1m = "/home/minhkv/datasets/feature/minhkv/sport1m"
feature_folder_sport1m_w = "/home/minhkv/feature/sport1m_rankpooling_w"

