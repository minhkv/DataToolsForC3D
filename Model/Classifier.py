from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
class Classifier:
    def __init__(self, classifier, train_file, test_file, class_ind, name=""):
        self.classifier = classifier
        self.train_file = train_file
        self.test_file = test_file
        self.class_ind = class_ind
        self.name = name
        self.train_input = []
        self.train_label = []
        self.test_input = []
        self.test_label = []
        self.test_pred = []
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.confusion_matrix = []

    def read_csv(self, filename):
        try:
            with open(filename, 'rb') as csvfile:
                feature=np.array([float(w) for w in csvfile.read().split(',')])
            return feature
        except IOError as er:
            print("[Error] IOError: {}".format(str(er)))
        
    def load_feature_from_csv(self, split_file):
        split_input = []
        split_label = []
        try:
            for csv_file, label in zip(split_file.name, split_file.label):
                feature = self.read_csv(csv_file)
                split_input.append(feature)
                split_label.append(label)
        except IOError as er:
            print("Error: {}".format(er))
            
        return split_input, split_label
    def load_train_test_split(self):
        print("[Info] Loading train test split {}".format(self.name))
        self.train_input, self.train_label = self.load_feature_from_csv(self.train_file)
        self.test_input, self.test_label = self.load_feature_from_csv(self.test_file)
    def training(self):
        print("[Info] Training classifier {} ".format(self.name))
        self.classifier.fit(self.train_input, self.train_label)
    def testing(self):
        print("[Info] Testing classifier {}".format(self.name))
        self.test_pred = self.classifier.predict(self.test_input)
        self.precision = precision_score(self.test_label, self.test_pred, average='macro')
        self.recall = recall_score(self.test_label, self.test_pred, average='macro')
        self.accuracy = accuracy_score(self.test_label, self.test_pred)
        labels = sorted(set(self.test_label))
        self.confusion_matrix = confusion_matrix(y_true=self.test_label, y_pred=self.test_pred, labels=range(len(self.class_ind.label)))
    def create_report(self):
        print("[Info] Creating report for classifier")
        report = classification_report(y_true=self.test_label, y_pred=self.test_pred, target_names=self.class_ind.name)
        with open('report{}.txt'.format(self.name), 'w') as fp:
            fp.write(report)

        np.savetxt(
            "confusion_matrix{}.csv".format(self.name),
            self.confusion_matrix,
            fmt='%.16f',
            delimiter=','
            )

        return