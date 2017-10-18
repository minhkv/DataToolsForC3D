import os
asset_path = os.path.abspath("Asset")
temp = os.path.abspath("Asset/tmp")
c3d_root = "/home/minhkv/C3D/C3D-v1.0/"
pre_trained="/home/minhkv/pre-trained/conv3d_deepnetA_sport1m_iter_1900000"
ucf101_video_folder="/home/minhkv/datasets/UCF101"
output_feature_folder = "/home/minhkv/feature"
train_file_line_syntax = r"(?P<name>.+) (?P<label>\w+)"
train_split_2_file_path = os.path.abspath("Asset/trainlist02.txt")
test_split_2_file_path = os.path.abspath("Asset/testlist02.txt")


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
