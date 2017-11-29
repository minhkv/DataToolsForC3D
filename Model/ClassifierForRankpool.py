from __future__ import print_function
import numpy as np
from Classifier import *
class ClassifierForRankpool(Classifier):
    def get_list_feature_in_folder(self, path, layer):
        listfiles = [path + '.csv']
        return listfiles
    def read_feature_file(self, filename):
        feature = self.read_csv(filename, sep='\n')
        feature_fow = feature[:4096]
        feature_rev = feature[-2 * 4096:-4096]
        # return feature[:4096]
        return np.concatenate((feature_fow, feature_rev))
        # return feature_fow