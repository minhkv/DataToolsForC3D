from config_global import *
flow_path = "/home/minhkv/datasets/ucf101_tvl1_flow/hmdb51_tvl1_flow"

# data_path = "/home/minhkv/datasets_old/hmdb51/data"
data_path = flow_path
stack_flow_path = "/home/minhkv/datasets_old/hmdb51/stack_tvl1_flow"
model_train = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_finetuning_train.prototxt")
model_test = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_finetuning_test.prototxt")
model_feature_extract = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_feature_extractor.prototxt")
solver = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_finetuning_solver.prototxt")
mean_split = os.path.join(temp, "mean_hmdb51_split_1.binaryproto")
# feature_folder_split = "/home/minhkv/feature/finetuned_hmdb51_split_1"
feature_folder_split = stack_flow_path

# pretrained_model = pre_trained_sport1m
pretrained_model = "/home/minhkv/pre-trained/hmdb51/c3d_hmdb51_finetune_iter_4000"
image = False

train_split_file_path = os.path.join(asset_path, "hmdb51_split", "train01.txt")
test_split_file_path = os.path.join(asset_path, "hmdb51_split", "test01.txt")

# train_split_file_path = os.path.join(asset_path, "hmdb51_split", "sample_train.txt")
# test_split_file_path = os.path.join(asset_path, "hmdb51_split", "sample_test.txt")

class_ind_path = os.path.join(asset_path, "hmdb51_split", "class_ind.txt")
