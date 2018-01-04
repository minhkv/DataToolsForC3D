from config_global import *

solver_train_ucf101 = os.path.join(asset_path, "conv3d_ucf101_solver.prototxt")
model_test_ucf101 = os.path.join(asset_path, "conv3d_ucf101_test.prototxt")
model_train_ucf101 = os.path.join(asset_path, "conv3d_ucf101_train.prototxt")
model_feature_extractor_ucf101 = os.path.join(asset_path, "conv3d_ucf101_feature_extractor.prototxt")

train_file_line_syntax = r"(?P<name>.+) (?P<label>\w+)"
test_file_line_syntax = r"(?P<label>.+)/(?P<name>.+)"
classInd_file_line_syntax = r"(?P<label>.+) (?P<name>\w+)"

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

ucf101_video_folder="/home/minhkv/datasets/UCF101"
ucf101_tvl1_flow_folder = "/home/minhkv/datasets/ucf101_tvl1_flow/ucf101_tvl1_flow"
ucf101_stack_tvl1_folder = "/home/minhkv/datasets/feature/tvl1_flow"

finetuned_ucf101_split1 = os.path.join(output_fine_tuned_net, "Split_1", "c3d_ucf101_finetune_whole_iter_20000")
finetuned_ucf101_split2 = os.path.join(output_fine_tuned_net, "Split_2", "c3d_ucf101_finetune_whole_iter_20000")
finetuned_ucf101_split3 = os.path.join(output_fine_tuned_net, "Split_3", "c3d_ucf101_finetune_whole_iter_20000")
feature_folder_ucf_split_1 = "/home/minhkv/datasets/feature/minhkv/finetuned_ucf101_split_1"
feature_folder_ucf_split_2 = "/home/minhkv/datasets/feature/minhkv/finetuned_ucf101_split_2"
feature_folder_ucf_split_3 = "/home/minhkv/datasets/feature/minhkv/finetuned_ucf101_split_3"

feature_folder_flow = "/home/minhkv/datasets/feature/minhkv/flow"

#config
data_path = ucf101_video_folder
model_train = model_train_ucf101
model_test = model_test_ucf101
model_feature_extract = model_feature_extractor_ucf101
solver = solver_train_ucf101
mean_split = mean_file_sport1m
feature_folder_split = feature_folder_ucf_split_1
pretrained_model = pre_trained_sport1m
image = False

class_ind_path = os.path.join(asset_path, "hmdb51_split", "class_ind.txt")
train_split_file_path = train_split_1_file_path
test_split_file_path = test_split_1_file_path
# train_split_file_path = os.path.join(asset_path, "hmdb51_split", "sample_train.txt")
# test_split_file_path = os.path.join(asset_path, "hmdb51_split", "sample_test.txt")
