import os
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

train_split_1_file_path = os.path.join(asset_path, "trainlist01.txt")
test_split_1_file_path = os.path.join(asset_path, "test_list01.txt")

train_split_2_file_path = os.path.join(asset_path, "trainlist02.txt")
test_split_2_file_path = os.path.join(asset_path, "test_list02.txt")

train_split_3_file_path = os.path.join(asset_path, "trainlist03.txt")
test_split_3_file_path = os.path.join(asset_path, "test_list03.txt")

classInd_file_path = os.path.join(asset_path, "classInd.txt")


ucf101_video_folder="/home/minhkv/datasets/UCF101"
pre_trained_sport1m="/home/minhkv/pre-trained/conv3d_deepnetA_sport1m_iter_1900000"
finetuned_ucf101_split2 = os.path.join(output_fine_tuned_net, "Split_2", "c3d_ucf101_finetune_whole_iter_20000")
finetuned_ucf101_split3 = os.path.join(output_fine_tuned_net, "Split_3", "c3d_ucf101_finetune_whole_iter_20000")

#  Change the following parameters for each split 
type_feature_file = "bin"
pretrained = finetuned_ucf101_split3
layer = "fc8-1"
mean_file = os.path.join(temp, "mean_split_3.binaryproto")
output_feature_folder = "/home/minhkv/feature/finetuned_ucf101_split_3"
train_split_file_path = train_split_3_file_path
test_split_file_path = test_split_3_file_path

classifier_name = "ucf_finetune_split_3"
clf = SVC(kernel="linear", C=0.025)


# For demo
# train_split_file_path = sample_train_file_path
# test_split_file_path = sample_test_file_path
# output_feature_folder = "/home/minhkv/feature/test"

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
