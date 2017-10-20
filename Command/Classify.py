from __future__ import print_function
from Command import *
import pandas as pd

class Classify(Command):
    def __init__(self, classfier):
        self.classifier = classfier
    def execute(self):
        self.classifier.load_train_test_split()
        self.classifier.training()
        self.classifier.testing()
        self.classifier.create_report()
        return