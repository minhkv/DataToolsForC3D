from config.config_global import *
from config.config_hmdb51 import *
# from config.config_ucf101 import *

#  Change the following parameters for each split 
c3d_root = "/home/minhkv/script/Run_C3D/C3D/C3D-v1.0" # for png image
use_image = image
type_image = "png" # training, finetune
input_folder_prefix = data_path # for training, finetune, test, feature extract
type_feature_file = "bin" # for classify
pretrained = pretrained_model # for feature extract, finetune, test
model_config = model_feature_extract # feature extract, train, test, finetune
solver_config = solver # train, finetune
layer = "fc6" # for convert and classify 
mean_file = mean_split # feature extract, finetune, test
output_feature_folder = feature_folder_split # for feature extract, classify, convert

classifier_name = "classifier_noname" # classify
if len(sys.argv) > 1:
    classifier_name = sys.argv[1]
# clf = SVC(kernel=additive_chi_square_kernel, C=0.005)
# feature_map = AdditiveChi2Sampler(sample_steps=2)
# clf = pipeline.Pipeline([("feature_map", feature_map), ("svm", LinearSVC())])
# clf.set_params(svm__C=0.01)

feature_map = MinMaxScaler(feature_range=(0, 100))
# feature_map = Normalizer()
estimator = SVC(kernel="linear", C=1) # classify
# estimator = SVC(kernel=additive_chi_square_kernel, C=0.01)
clf = pipeline.Pipeline([("feature_map", feature_map), ("svm", estimator)])
# clf = estimator
clf_precomputed = SVC(kernel="precomputed") # classify
