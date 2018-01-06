from config_global import *
flow_path = "/home/minhkv/datasets/ucf101_tvl1_flow/hmdb51_tvl1_flow"
stack_flow_path = "/home/minhkv/datasets_old/hmdb51/stack_tvl1_flow"

# data_path = flow_path # train, finetune

model_train_scratch = os.path.join(asset_path, "hmdb51_split", "conv3d_hmdb51_train.prototxt") 
model_test_scratch = os.path.join(asset_path, "hmdb51_split", "conv3d_hmdb51_test.prototxt") 
model_feature_extract_scratch = os.path.join(asset_path, "hmdb51_split", "conv3d_hmdb51_feature_extractor.prototxt") 
solver_train_scratch = os.path.join(asset_path, "hmdb51_split", "conv3d_hmdb51_solver.prototxt") 

model_flow_finetune_train = os.path.join(asset_path, "hmdb51_split", "conv3d_hmdb51_flow_finetune_train.prototxt") 

model_finetune_train = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_finetuning_train.prototxt") 
model_finetune_test = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_finetuning_test.prototxt")
model_finetune_feature_extract = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_feature_extractor.prototxt")
solver_finetune = os.path.join(asset_path, "hmdb51_split", "c3d_hmdb51_finetuning_solver.prototxt")

pretrained_flow = "/home/minhkv/pre-trained/hmdb51/conv3d_hmdb51_flow_s1_iter_20000"
pretrained_flow_ucf = "/home/minhkv/pre-trained/ucf101_flow/conv3d_ucf101_flow_s1_iter_60000"

data_path = "/home/minhkv/datasets_old/hmdb51/data"
data_path = stack_flow_path
model_train = model_flow_finetune_train
model_test = model_finetune_test
solver = solver_finetune
mean_split = os.path.join(temp, "mean_hmdb51_flow_split_1.binaryproto")
# feature_folder_split = "/home/minhkv/feature/finetuned_hmdb51_split_1" # classify, feature extract
feature_folder_split = stack_flow_path

# pretrained_model = pre_trained_sport1m
pretrained_model = pretrained_flow_ucf
# pretrained_model = "/home/minhkv/pre-trained/hmdb51/c3d_hmdb51_finetune_iter_4000"
image = True # train use_image

train_split_file_path = os.path.join(asset_path, "hmdb51_split", "train01.txt")
test_split_file_path = os.path.join(asset_path, "hmdb51_split", "test01.txt")

# train_split_file_path = os.path.join(asset_path, "hmdb51_split", "sample_train.txt") 
# test_split_file_path = os.path.join(asset_path, "hmdb51_split", "sample_test.txt")

# train_split_file_path = os.path.join(asset_path, "hmdb51_split", "lost_train.txt")
# test_split_file_path = os.path.join(asset_path, "hmdb51_split", "lost_test.txt")

class_ind_path = os.path.join(asset_path, "hmdb51_split", "class_ind.txt")
