from __future__ import print_function
from Command import *
import pandas as pd

class Classify(Command):
    def __init__(self, classfier, precomputed=False):
        self.classifier = classfier
        self.precomputed = precomputed
    def compute_chi_square(self):
        pass
    def execute(self):
        self.classifier.load_train_test_split()
        if self.precomputed:
            self.compute_chi_square()
        self.classifier.training()
        self.classifier.testing()
        self.classifier.create_report()
        return