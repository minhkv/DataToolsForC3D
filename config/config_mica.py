from config_global import *

solver_finetuning_mica = os.path.join(asset_path, "c3d_mica_finetuning_solver.prototxt")
model_finetune_mica_train = os.path.join(asset_path, "c3d_mica_finetuning_train.prototxt")
model_finetune_mica_test = os.path.join(asset_path, "c3d_mica_finetuning_test.prototxt")
model_finetune_mica_feature_extractor = os.path.join(asset_path, "c3d_mica_feature_extractor.prototxt")

feature_folder_mica = "/home/minhkv/datasets/feature/minhkv/mica_4000"
feature_folder_mica_dense = "/home/minhkv/datasets/feature/minhkv/mica_dense"
feature_folder_mica_dense_rankpool = "/home/minhkv/feature/minhkv/Rankpool/mica_4000_dense"
feature_folder_mica_rankpool = "/home/minhkv/feature/minhkv/Rankpool/mica_4000"
feature_folder_mica_v5 = "/home/minhkv/datasets/feature/minhkv/mica_3200_v5_re"
feature_folder_mica_v4 = "/home/minhkv/datasets/feature/minhkv/mica_2400_v4"
feature_folder_mica_sport1m = "/home/minhkv/datasets/feature/minhkv/mica_sport1m_v5"
mica_annotation_path = "/home/minhkv/datasets/Kinect_vis_annotate/v5/Kinect3" # classify
mica_annotation_mapping_path = os.path.join(mica_annotation_path, "file_mapping.txt")
mica_video_path = "/home/minhkv/datasets/Kinect2017-10/Datasets"
test_mica_mapping = os.path.join(asset_path, "file_mapping_test.txt")
# pretrained_mica = "/home/minhkv/pre-trained/mica/c3d_MICA_less_finetune_iter_3600"
pretrained_mica = "/home/minhkv/pre-trained/mica_fix_label/c3d_MICA_finetune_iter_4000"

mica_split_syntax = r"(?P<label>.+) (?P<start_in_video>.+) (?P<end_in_video>\w+) (?P<name>.+) (?P<order>.+)"
mica_train_path = os.path.join(asset_path, 'mica_train.txt')
mica_test_path = os.path.join(asset_path, 'mica_test.txt')
