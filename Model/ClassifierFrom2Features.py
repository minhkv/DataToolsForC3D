from __future__ import print_function
from Classifier import *

class ClassifierFrom2Features(Classifier):
    def combine_list_feature(self, list_feature):
        num_feature = len(list_feature)
        print ("[DEBUG] List feature len: {}".format(num_feature))
        list_feature = sorted(list_feature)
        mid = num_feature / 2
        feature = self.read_feature_file(list_feature[mid])
        if mid == 0:
            feature = feature * 2
        else:
            feature = self.read_feature_file(list_feature[mid - 1]) + feature
        return feature