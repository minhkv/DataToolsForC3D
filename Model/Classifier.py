from __future__ import print_function
import numpy as np
import pandas as pd
import glob
import os
import sys
import array
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.manifold import TSNE
from sklearn import pipeline
import pickle
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
    def read_csv(self, filename, sep=','):
        with open(filename, 'rb') as csvfile:
            feature=np.array([float(w) for w in csvfile.read().split(sep) if w])
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
        listfiles = sorted([os.path.join(path, f) for f in os.listdir(path) if layer in f])
        return listfiles

    def combine_list_feature(self, list_feature):
        if len(list_feature) == 0:
            return np.zeros(20)
        features = [self.read_feature_file(csv_file) for csv_file in list_feature]
        feature = np.mean(features, axis=0)
        return feature

    def load_feature_from_folder_and_average(self, split_file):
        split_input = []
        split_label = []
        try:
            for feature_folder, label in zip(split_file.name, split_file.label):
                print("[Load] Loading: {} {}".format(feature_folder, label))
                list_feature = self.get_list_feature_in_folder(feature_folder, self.layer)
                if len(list_feature) == 0:
                    self.empty_folder.append(feature_folder)
                    # continue
                self.test_name.append(feature_folder)
                features = self.combine_list_feature(list_feature)
                split_input.append(features)
                split_label.append(label)
        except IOError as er:
            print("[Error] Error: {}".format(er))
        return split_input, split_label

    def load_train_test_split(self):
        print("[Classifier] Loading train test split {}".format(self.name))
        self.train_input, self.train_label = self.load_feature_from_folder_and_average(self.train_file)
        self.test_input, self.test_label = self.load_feature_from_folder_and_average(self.test_file)
        print ("[Classifier] Loaded {} train feature".format(len(self.train_label)))
        print ("[Classifier] Loaded {} test feature".format(len(self.test_label)))

    def transform_data(self):
        pass
        
    def fuse_with(self, clf, fuse_func=None):
        """Fusion 2 classifier after load train_input
        fuse_func: take two input vector and produce output fusion vector
        """
        if (len(self.train_input) != len(clf.train_input)):
            raise ValueError("2 train_input not have same length")
        if fuse_func != None:
            self.train_input = [fuse_func(input1, input2) for input1, input2 in zip(self.train_input, clf.train_input)]
            self.test_input = [fuse_func(input1, input2) for input1, input2 in zip(self.test_input, clf.test_input)]

    def training(self):
        print("[Classifier] Training classifier {} ".format(self.name))
        self.classifier.fit(self.train_input, self.train_label)

    def testing(self):
        print("[Classifier] Testing classifier {}".format(self.name))
        self.test_pred = self.predict(self.test_input)
        self.confusion_matrix = confusion_matrix(
            y_true=self.test_label,
            y_pred=self.test_pred, 
            labels=range(len(self.class_ind.label))
            )
        print(' '.join(self.class_ind.label))
        t = (set(self.test_label))
        print (t)
        p = (set(self.test_pred))
        print (p)
    
    def gen_test_proba(self):
        print("[Classifier] Generating prob {}".format(self.name))
        self.train_input = self.predict_proba(self.train_input)
        self.test_input = self.predict_proba(self.test_input)
    def testing_proba(self):
        print("[Classifier] Testing classifier using prob predict {}".format(self.name))
        self.test_pred = [np.argmax(p) for p in self.test_input]
        self.confusion_matrix = confusion_matrix(
            y_true=self.test_label,
            y_pred=self.test_pred, 
            labels=range(len(self.class_ind.label))
            )
        print(' '.join(self.class_ind.label))
        t = (set(self.test_label))
        print (t)
        p = (set(self.test_pred))
        print (p)

    def predict(self, x_test):
        return self.classifier.predict(x_test)
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    def save_tsne(self, feature_input, input_label, filename):
        feature_map = Normalizer()
        tsne = TSNE(n_components=2)
        mapping = pipeline.Pipeline([("feature_map", feature_map), ("tsne", tsne)])
        tsne_feature = mapping.fit_transform(feature_input)
        with open(filename, "w") as tt:
            content = []
            for item, label in zip(tsne_feature, input_label):
                content.append("{} {} {}".format(label, item[0], item[1]))
            tt.write('\n'.join(content))

    def visualize_feature(self):
        train_tsne = "train_tsne_{}.csv".format(self.name)
        test_tsne = "test_tsne_{}.csv".format(self.name)
        print("[Visualize] Processing train input")
        self.save_tsne(self.train_input, self.train_label, train_tsne)
        print("[Visualize] Processing test input")
        self.save_tsne(self.test_input, self.test_label, test_tsne)
        
    def save_classifier(self):
        pickle.dump(self.classifier, self.name + ".pkl")
    def save_output(self):
        pass
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