from config_global import *

class_ind_path = "/media/data/datasets/Kinect_vis_annotate/ActionIndex.txt"
mica_split_syntax = r"(?P<label>.+) (?P<start_in_video>.+) (?P<end_in_video>\w+) (?P<name>.+) (?P<order>.+)"
mica_train_path = os.path.join(asset_path, 'mica_split', 'mica_train.txt')
mica_test_path = os.path.join(asset_path, 'mica_split', 'mica_test.txt')

model_train = "/home/minhkv/script/MICA_C3D/DataToolsForC3D/Asset/mica_split/conv3d_mica_train.prototxt"
model_test = "/home/minhkv/script/MICA_C3D/DataToolsForC3D/Asset/mica_split/conv3d_mica_test.prototxt"
model_feature_extract = "/home/minhkv/script/MICA_C3D/DataToolsForC3D/Asset/mica_split/conv3d_mica_feature_extractor.prototxt"
solver = "/home/minhkv/script/MICA_C3D/DataToolsForC3D/Asset/mica_split/conv3d_mica_solver.prototxt"

model_flow_train = os.path.join(asset_path, "mica_split", "conv3d_mica_train.prototxt")
model_flow_test = os.path.join(asset_path, "mica_split", "conv3d_mica_test.prototxt")
model_flow_feature_extract = os.path.join(asset_path, "mica_split", "conv3d_mica_feature_extractor.prototxt")
mean_file_flow = os.path.join(temp, "mean_mica_flow.binaryproto")

# depth_path = "/home/minhkv/datasets/Kinect_clips/depth"
depth_path = "/home/minhkv/datasets_old/mica/depth"
mean_file_depth = os.path.join(temp, "mean_mica_depth.binaryproto")

rgb_feature_folder = "/home/minhkv/datasets/feature/minhkv/mica_4000"
flow_feature_folder = "/home/minhkv/feature/mica/merged_flow"

stack_flow_mica = "/home/minhkv/datasets_old/mica/flow/merged_flow"

sample_train = os.path.join(asset_path, 'mica_split', 'sample_train.txt')
sample_test = os.path.join(asset_path, 'mica_split', 'sample_test.txt')
file_mapping = "/home/minhkv/datasets/Kinect_vis_annotate/v5/Kinect3/file_mapping.txt"
image = True