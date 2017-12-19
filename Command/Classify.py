from __future__ import print_function
from Command import *
import pandas as pd

class Classify(Command):
    def __init__(self, classifier, classifier2=None, fuse_func=None):
        self.classifier = classifier
        self.classifier2 = classifier2
        self.fuse_func = fuse_func
    def execute(self):
        self.classifier.load_train_test_split()
        self.classifier.transform_data()
        if self.classifier2 != None and self.fuse_func != None:            
            self.classifier2.load_train_test_split()
            self.classifier.transform_data()
            self.classifier.fuse_with(self.classifier2, fuse_func=self.fuse_func)
        self.classifier.training()
        self.classifier.testing()
        self.classifier.create_report()
        return