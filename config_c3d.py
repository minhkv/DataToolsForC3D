from config.config_global import *
# from config.config_hmdb51 import *
# from config.config_ucf101 import *
from config.config_mica import *

#  Change the following parameters for each split 
# c3d_root = "/home/minhkv/script/Run_C3D/C3D/C3D-v1.0" # for png image
c3d_root = c3d_mica_flow # for mica depth png image
use_image = image
type_image = "png" # training, finetune
type_feature_file = "bin" # for classify
layer = "prob" # for convert and classify 

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
