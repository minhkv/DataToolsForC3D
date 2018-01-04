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
c3d_root = "/home/minhkv/C3D/C3D-v1.0/"

model_feature_extractor_sport1m = os.path.join(asset_path, "c3d_sport1m_feature_extractor.prototxt")

solver_train_ucf101 = os.path.join(asset_path, "conv3d_ucf101_solver.prototxt")
model_test_ucf101 = os.path.join(asset_path, "conv3d_ucf101_test.prototxt")
model_train_ucf101 = os.path.join(asset_path, "conv3d_ucf101_train.prototxt")
model_feature_extractor_ucf101 = os.path.join(asset_path, "conv3d_ucf101_feature_extractor.prototxt")

solver_finetuning_mica = os.path.join(asset_path, "c3d_mica_finetuning_solver.prototxt")
model_finetune_mica_train = os.path.join(asset_path, "c3d_mica_finetuning_train.prototxt")
model_finetune_mica_test = os.path.join(asset_path, "c3d_mica_finetuning_test.prototxt")
model_finetune_mica_feature_extractor = os.path.join(asset_path, "c3d_mica_feature_extractor.prototxt")

train_file_line_syntax = r"(?P<name>.+) (?P<label>\w+)"
test_file_line_syntax = r"(?P<label>.+)/(?P<name>.+)"
classInd_file_line_syntax = r"(?P<label>.+) (?P<name>\w+)"

input_chunk_list_line_syntax = "{0} {1} {2}\n"
output_chunk_list_line_syntax = "{0}/{1:06d}\n"

input_chunk_file = os.path.join(temp, "input.txt")
output_chunk_file = os.path.join(temp, "output.txt")

empty_split_file_path = os.path.join(asset_path, "empty.txt")
sample_train_file_path = os.path.join(asset_path, "sample_train.txt")
sample_test_file_path = os.path.join(asset_path, "sample_test.txt")

lost_train_file_path = os.path.join(asset_path, "lost_train.txt")
lost_test_file_path = os.path.join(asset_path, "lost_test.txt")

train_split_1_file_path = os.path.join(asset_path, "trainlist01.txt")
test_split_1_file_path = os.path.join(asset_path, "test_list01.txt")

train_split_2_file_path = os.path.join(asset_path, "trainlist02.txt")
test_split_2_file_path = os.path.join(asset_path, "test_list02.txt")

train_split_3_file_path = os.path.join(asset_path, "trainlist03.txt")
test_split_3_file_path = os.path.join(asset_path, "test_list03.txt")

classInd_file_path = os.path.join(asset_path, "classInd.txt")
mean_file_ucf_split_1 = os.path.join(temp, "mean_split_1.binaryproto")
mean_file_ucf_split_2 = os.path.join(temp, "mean_split_2.binaryproto")
mean_file_ucf_split_3 = os.path.join(temp, "mean_split_3.binaryproto")
mean_file_sport1m = os.path.join(temp, "sport1m_train16_128_mean.binaryproto")

mean_file_ucf_opt_flow_split_1 = os.path.join(temp, "mean_opt_flow_split_1.binaryproto")
mean_mica_split = os.path.join(temp, "mean_mica_split.binaryproto")
report_folder = os.path.abspath("report")

ucf101_video_folder="/home/minhkv/datasets/UCF101"
ucf101_tvl1_flow_folder = "/home/minhkv/datasets/ucf101_tvl1_flow/ucf101_tvl1_flow"
ucf101_stack_tvl1_folder = "/home/minhkv/datasets/feature/tvl1_flow"
pre_trained_sport1m="/home/minhkv/pre-trained/conv3d_deepnetA_sport1m_iter_1900000"
finetuned_ucf101_split1 = os.path.join(output_fine_tuned_net, "Split_1", "c3d_ucf101_finetune_whole_iter_20000")
finetuned_ucf101_split2 = os.path.join(output_fine_tuned_net, "Split_2", "c3d_ucf101_finetune_whole_iter_20000")
finetuned_ucf101_split3 = os.path.join(output_fine_tuned_net, "Split_3", "c3d_ucf101_finetune_whole_iter_20000")
feature_folder_ucf_split_1 = "/home/minhkv/datasets/feature/minhkv/finetuned_ucf101_split_1"
feature_folder_ucf_split_2 = "/home/minhkv/datasets/feature/minhkv/finetuned_ucf101_split_2"
feature_folder_ucf_split_3 = "/home/minhkv/datasets/feature/minhkv/finetuned_ucf101_split_3"
feature_folder_sport1m = "/home/minhkv/datasets/feature/minhkv/sport1m"
feature_folder_sport1m_dense = "/home/minhkv/feature/minhkv/sport1m_dense"
feature_folder_flow = "/home/minhkv/datasets/feature/minhkv/flow"
feature_folder_mica = "/home/minhkv/datasets/feature/minhkv/mica_4000"
feature_folder_mica_dense = "/home/minhkv/datasets/feature/minhkv/mica_dense"
feature_folder_mica_dense_rankpool = "/home/minhkv/feature/minhkv/Rankpool/mica_4000_dense"
feature_folder_mica_rankpool = "/home/minhkv/feature/minhkv/Rankpool/mica_4000"
feature_folder_mica_v5 = "/home/minhkv/datasets/feature/minhkv/mica_3200_v5_re"
feature_folder_mica_v4 = "/home/minhkv/datasets/feature/minhkv/mica_2400_v4"
feature_folder_mica_sport1m = "/home/minhkv/datasets/feature/minhkv/mica_sport1m_v5"
# w_2 for 2nd kernel
feature_folder_sport1m_w = "/home/minhkv/feature/sport1m_rankpooling_w"

mica_annotation_path = "/home/minhkv/datasets/Kinect_vis_annotate/v5/Kinect3" # classify
# mica_annotation_path = "/home/minhkv/datasets/Kinect_vis_annotate/v4/Kinect3" # classify

mica_annotation_mapping_path = os.path.join(mica_annotation_path, "file_mapping.txt")
mica_video_path = "/home/minhkv/datasets/Kinect2017-10/Datasets"
test_mica_mapping = os.path.join(asset_path, "file_mapping_test.txt")
# pretrained_mica = "/home/minhkv/pre-trained/mica/c3d_MICA_less_finetune_iter_3600"
pretrained_mica = "/home/minhkv/pre-trained/mica_fix_label/c3d_MICA_finetune_iter_4000"

mica_split_syntax = r"(?P<label>.+) (?P<start_in_video>.+) (?P<end_in_video>\w+) (?P<name>.+) (?P<order>.+)"
mica_train_path = os.path.join(asset_path, 'mica_train.txt')
mica_test_path = os.path.join(asset_path, 'mica_test.txt')

#  Change the following parameters for each split 
c3d_root = "/home/minhkv/script/Run_C3D/C3D/C3D-v1.0" # for png image
use_image = False
type_image = "png" # training, finetune
input_folder_prefix = mica_video_path # for training, finetune, test, feature extract
type_feature_file = "bin" # for classify
pretrained = pretrained_mica # for feature extract, finetune, test
model_config = model_finetune_mica_feature_extractor # feature extract, train, test, finetune
solver_config = solver_finetuning_mica # train, finetune
layer = "prob" # for convert and classify 
mean_file = mean_mica_split # feature extract, finetune, test
output_feature_folder = feature_folder_mica # for feature extract, classify, convert

# only ucf101
train_split_file_path = train_split_1_file_path # finetune, feature extract, classify, convert, test
test_split_file_path = test_split_1_file_path # finetune, feature extract, classify, convert, test

classifier_name = "classifier_noname" # classify
if len(sys.argv) > 1:
    classifier_name = sys.argv[1]
# clf = SVC(kernel=additive_chi_square_kernel, C=0.005)
# feature_map = AdditiveChi2Sampler(sample_steps=2)
# clf = pipeline.Pipeline([("feature_map", feature_map), ("svm", LinearSVC())])
# clf.set_params(svm__C=0.01)

feature_map = MinMaxScaler(feature_range=(0, 100))
# feature_map = Normalizer()
# estimator = SVC(kernel="linear", C=0.01) # classify
estimator = SVC(kernel=additive_chi_square_kernel, C=0.01)
clf = pipeline.Pipeline([("feature_map", feature_map), ("svm", estimator)])
# clf = estimator
clf_precomputed = SVC(kernel="precomputed") # classify

# For lost file
# train_split_file_path = lost_train_file_path
# test_split_file_path = lost_test_file_path
# train_split_file_path = empty_split_file_path
# test_split_file_path = "/home/minhkv/script/DataToolsForC3D/Asset/analyse_w_list.txt"

# ucf101_stack_tvl1_folder = "/home/minhkv/datasets/feature/tvl1_flow_test"
# input_folder_prefix = ucf101_stack_tvl1_folder
# For demo
train_split_file_path = sample_train_file_path
test_split_file_path = sample_test_file_path
# output_feature_folder = "/home/minhkv/feature/test"
# classifier_name = "classifier_test"

"""
Parameters for model.prototxt:
- Training, Finetuning: shuffle = True, mirror = True, use_temporal_jitter = False
- Testing: shuffle = False, mirror = False, use_temporal_jitter = False
- Feature extraction: shuffle = False, mirror = False, use_temporal_jitter = None

** If using video: use_image = False. Otherwise use_image = True
"""
# shuffle = True 
# mirror = True
# batch_size = 20
# use_temporal_jitter = False
# length = 16
# height = 128
# width = 171
# use_image
