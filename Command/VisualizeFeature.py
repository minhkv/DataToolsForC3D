from __future__ import print_function
from Command import *

class VisualizeFeature(Command):
    """
        Use t-sne to reduce feature to 2D space
        syntax: train_input(or test_input) = [t-sne feature] (t-sne feature has 2d)
                label = [label]
    """
    def __init__(self, classifier):
        self.classifier = classifier
    def execute(self):
        self.classifier.load_train_test_split()
        self.classifier.transform_data()
        self.classifier.visualize_feature()