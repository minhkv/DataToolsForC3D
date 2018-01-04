from __future__ import print_function
import numpy as np
from Classifier import *
class ClassifierClipProb(Classifier):
	
	def training(self):
		return
	def predict(self, x_test):
		return np.argmax(x_test)
	def testing(self): 
		print ("[Info] Testing")
		self.test_pred = [self.predict(x_test) for x_test in self.test_input]
		self.confusion_matrix = confusion_matrix(
			y_true=self.test_label,
			y_pred=self.test_pred, 
			labels=range(len(self.class_ind.label))
		)
	def combine_list_feature(self, list_feature):
		features = [self.read_feature_file(csv_file) for csv_file in list_feature]
		# feature = np.mean(features, axis=0)
		# print("[Combine] Folder: {}".format(os.path.dirname(list_feature[0])))
		# print("[Combine] List features: {}".format(np.array(features).shape))
		
		return features
	def load_feature_from_folder_and_average(self, split_file):
		split_input = []
		split_label = []
		try:
			for feature_folder, label in zip(split_file.name, split_file.label):
				print("[Load] Loading prob: {} {}".format(feature_folder, label))
				list_feature = self.get_list_feature_in_folder(feature_folder, self.layer)
				if len(list_feature) == 0:
					self.empty_folder.append(feature_folder)
					continue
				self.test_name.append(feature_folder)
				features = self.combine_list_feature(list_feature)
				# print("[Combine] Feature combined: {}".format(np.array(features).shape))
				split_input.extend(features)
				for i in range(len(list_feature)):
					split_label.append(label)
		except IOError as er:
			print("[Error] Error: {}".format(er))
		print("[Load classify] Input shape: {}, label shape: {}".format(np.array(split_input).shape, np.array(split_label).shape))
		return split_input, split_label