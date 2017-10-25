import os
from sklearn.svm import SVC
asset_path = os.path.abspath("Asset")
temp = os.path.abspath("Asset/tmp")
fine_tuned_net_folder = os.path.abspath("Finetuned_net")
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

#  Change the following parameters for each split 
ucf101_video_folder="/home/minhkv/datasets/UCF101"
pre_trained="/home/minhkv/pre-trained/conv3d_deepnetA_sport1m_iter_1900000"
type_feature_file = "csv"
output_fine_tuned_net = os.path.abspath("Finetuned_net")

layer = "fc6-1"
output_feature_folder = "/home/minhkv/feature/finetuned_ucf101_split_2"
train_split_file_path = train_split_2_file_path
test_split_file_path = test_split_2_file_path
classifier_name = "ucf_finetune_split_2"
clf = SVC(kernel="linear", C=0.025)

# train_split_file_path = sample_train_file_path
# test_split_file_path = sample_test_file_path

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
