from __future__ import print_function
from Command import *
import numpy as np

class ClassifyEarlyFusion(Command):
    def __init__(self, classifier, classifier2=None, fuse_func=None, save=False):
        self.classifier = classifier
        self.classifier2 = classifier2
        self.fuse_func = fuse_func
        self.save = save
    def execute(self):
        print("[Classify] Stream 1")
        self.classifier.load_train_test_split()
        self.classifier.transform_data()
        print("[Classify] Stream 2")
        self.classifier2.load_train_test_split()
        self.classifier2.transform_data()
        print("[Classify] Fusing feature vector")
        self.classifier.fuse_with(self.classifier2, fuse_func=self.fuse_func)
        print("[Classify] Train data shape: {}".format(np.array(self.classifier.train_input).shape))
        print("[Classify] Test data shape: {}".format(np.array(self.classifier.test_input).shape))
        self.classifier.training()
        self.classifier.testing()
        self.classifier.create_report()
        return
    