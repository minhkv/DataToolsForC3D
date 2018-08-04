from __future__ import print_function
from Command import *
import numpy as np

class Classify(Command):
    def __init__(self, classifier, classifier2=None, fuse_func=None, save=False):
        self.classifier = classifier
        self.classifier2 = classifier2
        self.fuse_func = fuse_func
        self.save = save
    def execute(self):
        self.classifier.load_train_test_split()
        self.classifier.transform_data()
        if self.classifier2 != None and self.fuse_func != None:            
            self.classifier2.load_train_test_split()
            self.classifier2.transform_data()
            self.classifier.fuse_with(self.classifier2, fuse_func=self.fuse_func)
        print("[Classify] Train data shape: {}".format(np.array(self.classifier.train_input).shape))
        if len(self.classifier.train_input) > 0:
            print("[Classify] Train data min value: {}".format(np.array(self.classifier.train_input).min()))
            print("[Classify] Train data max value: {}".format(np.array(self.classifier.train_input).max()))

        print("[Classify] Test data shape: {}".format(np.array(self.classifier.test_input).shape))
        if len(self.classifier.test_input) > 0:
            print("[Classify] Test data min value: {}".format(np.array(self.classifier.test_input).min()))
            print("[Classify] Test data max value: {}".format(np.array(self.classifier.test_input).max()))
        self.classifier.training()
        self.classifier.testing()
        self.classifier.create_report()
        self.classifier.save_output()
        # if self.save:
        #     print("[Classify] Saving classifier: {}".format(self.classifier.name))
        #     self.classifier.save_classifier()
        return