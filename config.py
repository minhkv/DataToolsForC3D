import os
import sys
from sklearn.svm import SVC
asset_path = os.path.abspath("Asset")
temp = os.path.abspath("Asset/tmp")
output_fine_tuned_net = os.path.abspath("Finetuned_net")
c3d_root = "/home/minhkv/C3D/C3D-v1.0/"
train_file_line_syntax = r"(?P<name>.+) (?P<label>\w+)"
test_file_line_syntax = r"(?P<label>.+)/(?P<name>.+)"
classInd_file_line_syntax = r"(?P<label>.+) (?P<name>\w+)"

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


ucf101_video_folder="/home/minhkv/datasets/UCF101"
pre_trained_sport1m="/home/minhkv/pre-trained/conv3d_deepnetA_sport1m_iter_1900000"
finetuned_ucf101_split1 = os.path.join(output_fine_tuned_net, "Split_1", "c3d_ucf101_finetune_whole_iter_20000")
finetuned_ucf101_split2 = os.path.join(output_fine_tuned_net, "Split_2", "c3d_ucf101_finetune_whole_iter_20000")
finetuned_ucf101_split3 = os.path.join(output_fine_tuned_net, "Split_3", "c3d_ucf101_finetune_whole_iter_20000")
feature_folder_ucf_split_1 = "/home/minhkv/feature/finetuned_ucf101_split_1"
feature_folder_ucf_split_2 = "/home/minhkv/feature/finetuned_ucf101_split_2"
feature_folder_ucf_split_3 = "/home/minhkv/feature/finetuned_ucf101_split_3"
feature_folder_sport1m = "/home/minhkv/feature/sport1m"

#  Change the following parameters for each split 
type_feature_file = "bin" # for classify
pretrained = finetuned_ucf101_split3 # for feature extraction, finetune, test
layer = "fc6" # for converting and classify
mean_file = mean_file_ucf_split_3 # feature extract, finetune, test
output_feature_folder = feature_folder_ucf_split_1 # for feature extract, classify
train_split_file_path = train_split_1_file_path # finetune, feature extract, classify, convert, test
test_split_file_path = test_split_1_file_path # finetune, feature extract, classify, convert, test

classifier_name = "classifier_noname" # classify
if len(sys.argv) > 1:
    classifier_name = sys.argv[1]
clf = SVC(kernel="linear", C=0.025) # classify

# For lost file
# train_split_file_path = lost_train_file_path
# test_split_file_path = lost_test_file_path

# For demo
# train_split_file_path = sample_train_file_path
# test_split_file_path = sample_test_file_path
# output_feature_folder = "/home/minhkv/feature/test"
# classifier_name = "classifier_test"

"""
Parameters for model.prototxt:
- Training, Finetuning: shuffle = True, mirror = True, use_temporal_jitter = False
- Testing: shuffle = False, mirror = False, use_temporal_jitter = False
- Feature extraction: shuffle = False, mirror = False, use_temporal_jitter = None

** If using video: use_image = False. Otherwise use_image = False
"""
# shuffle = True 
# mirror = True
# batch_size = 20
# use_temporal_jitter = False
# length = 16
# height = 128
# width = 171
# use_image
