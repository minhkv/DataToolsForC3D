from __future__ import print_function
import numpy as np
import pandas as pd
import glob
import os
import sys
import array
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
class Classifier:
    def __init__(
        self, 
        train_file, 
        test_file, 
        class_ind, 
        classifier=None, 
        name="", 
        layer="fc6-1",
        type_feature_file="csv"):
        self.classifier = classifier
        self.train_file = train_file
        self.test_file = test_file
        self.class_ind = class_ind
        self.name = name
        self.train_input = []
        self.train_label = []
        self.test_name = []
        self.test_input = []
        self.test_label = []
        self.test_pred = []
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.confusion_matrix = []
        self.empty_folder = []
        self.layer = layer
        self.type_feature_file = type_feature_file

    def read_bin(self, filename):
		with open(filename, 'rb') as f:
			s = f.read()
			(n, c, l, h, w) = array.array("i", s[:20])
			feature_vec = array.array("f", s[20:])
			if not self.layer == "prob":
				feature_vec = np.array(array.array("f", [float("{:16f}".format(i)) for i in feature_vec]))
		return feature_vec
    def read_csv(self, filename):
        with open(filename, 'rb') as csvfile:
            feature=np.array([float(w) for w in csvfile.read().split(',')])
            return feature
    def read_feature_file(self, filename):
        if(self.type_feature_file == "bin"):
            feature = self.read_bin(filename)
        elif (self.type_feature_file == "csv"):
            feature = self.read_csv(filename)
        else: 
            print("[Error] Not have feature filetype: {}".format(self.type_feature_file))
            sys.exit(-6)
        return feature

    def get_list_feature_in_folder(self, path, layer):
        listfiles = glob.glob(os.path.join(path, "*" + layer + "*"))
        return listfiles

    def combine_list_feature(self, list_feature):
        features = [self.read_feature_file(csv_file) for csv_file in list_feature]
        feature = np.mean(features, axis=0)
        return feature

    def load_feature_from_folder_and_average(self, split_file):
        split_input = []
        split_label = []
        try:
            for feature_folder, label in zip(split_file.name, split_file.label):
                print("[Info] Loading feature from: {}".format(os.path.basename(feature_folder)))
                list_feature = self.get_list_feature_in_folder(feature_folder, self.layer)
                if len(list_feature) == 0:
                    class_name = self.class_ind.convert_label_to_name(str(label + 1))
                    input_name = os.path.basename(feature_folder) + ".avi"
                    self.empty_folder.append(os.path.join(class_name, input_name))
                    continue
                self.test_name.append(feature_folder)
                features = self.combine_list_feature(list_feature)
                split_input.append(features)
                split_label.append(label)
        except IOError as er:
            print("[Error] Error: {}".format(er))
            
        return split_input, split_label
    def load_train_test_split(self):
        print("[Info] Loading train test split {}".format(self.name))
        self.train_input, self.train_label = self.load_feature_from_folder_and_average(self.train_file)
        self.test_input, self.test_label = self.load_feature_from_folder_and_average(self.test_file)
        print ("[Info] Loaded {} train feature".format(len(self.train_label)))
        print ("[Info] Loaded {} test feature".format(len(self.test_label)))
    def training(self):
        print("[Info] Training classifier {} ".format(self.name))
        self.classifier.fit(self.train_input, self.train_label)
    def testing(self):
        print("[Info] Testing classifier {}".format(self.name))
        self.test_pred = self.classifier.predict(self.test_input)
        # self.precision = precision_score(self.test_label, self.test_pred, average='macro')
        # self.recall = recall_score(self.test_label, self.test_pred, average='macro')
        # self.accuracy = accuracy_score(self.test_label, self.test_pred)
        self.confusion_matrix = confusion_matrix(
            y_true=self.test_label,
            y_pred=self.test_pred, 
            labels=range(len(self.class_ind.label))
            )
    def create_report(self):
        print("[Info] Creating report for classifier")
        report = classification_report(
            y_true=self.test_label, 
            y_pred=self.test_pred, 
            target_names=self.class_ind.name,
            digits=7)
        with open('report{}.txt'.format(self.name), 'w') as fp:
            fp.write(report)
        np.savetxt(
            "test_name_{}.csv".format(self.name),
            self.test_name,
            fmt='%s',
            delimiter=','
            )
        np.savetxt(
            "y_true_{}.csv".format(self.name),
            self.test_label,
            fmt='%d',
            delimiter=','
            )
        np.savetxt(
            "y_pred_{}.csv".format(self.name),
            self.test_pred,
            fmt='%d',
            delimiter=','
            )
        np.savetxt(
            "confusion_matrix_{}.csv".format(self.name),
            self.confusion_matrix,
            fmt='%5d',
            delimiter=','
            )
        np.savetxt(
            "empty_folder_{}.csv".format(self.name),
            self.empty_folder,
            fmt='%s',
            delimiter=','
            )

        return