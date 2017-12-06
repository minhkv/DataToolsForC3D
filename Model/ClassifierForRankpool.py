from __future__ import print_function
import numpy as np
from Classifier import *
# from sklearn.preprocessing import normalize
class ClassifierForRankpool(Classifier):
    def get_list_feature_in_folder(self, path, layer):
        listfiles = [path + '.csv']
        return listfiles
    def read_feature_file(self, filename):
        feature = self.read_csv(filename, sep='\n')
        feature_fow = feature[:4096]
        feature_rev = feature[-2 * 4096:-4096]
        feature = np.concatenate((feature_fow, feature_rev))
        # n = (np.linalg.norm(feature))
        # if n < 1:
        #     print (n)
        # if n > 0:
        #     feature = feature * 10 / float(np.linalg.norm(feature))
        
        return feature
    # def combine_list_feature(self, list_feature):
    #     pass