from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
class Classfier:
    def __init__(self, classifier, train_file, test_file):
        self.classifier = classifier
        self.train_file = train_file
        self.test_file = test_file
        self.train_input = []
        self.train_label = []
        self.test_input = []
        self.test_label = []
        self.test_pred = []
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.confusion_matrix = []
    def load_feature_from_csv(self, split_file):
        split_input = []
        split_label = []
        for csv_file, label in zip(split_file.name, split_file.label):
            feature = pd.read_csv(csv_file)
            split_input.append(feature)
            split_label.append(label)
        return split_input, split_label
    def load_train_test_split(self):
        self.train_input, self.train_label = self.load_feature_from_csv(self.train_file)
        self.test_input, self.test_label = self.load_feature_from_csv(self.test_file)
    def training(self):
        self.classifier.fit(self.train_input, self.train_label)
    def testing(self):
        self.test_pred = self.classifier.predict(self.test_input)
        self.precision = precision_score(self.test_label, self.test_pred, average='macro')
        self.recall = recall_score(self.test_label, self.test_pred, average='macro')
        self.accuracy = accuracy_score(self.test_label, self.test_pred)
        self.confusion_matrix = confusion_matrix(self.test_label, self.test_pred, labels=[i for i in range(101)])

    def create_report(self):
        return