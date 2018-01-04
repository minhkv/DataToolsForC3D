from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Classifier import *
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp


class ClassifierUsingProb(Classifier):
	
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
		print(' '.join(self.class_ind.label))
		t = (set(self.test_label))
		print (t)
		p = (set(self.test_pred))
		print (p)
	def save_output(self):
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		n_classes = len(self.class_ind.label)
		y_test = label_binarize(self.test_label, classes=range(n_classes))
		y_score = np.array(self.test_input)
		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])

		# Compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes):
			mean_tpr += interp(all_fpr, fpr[i], tpr[i])

		# Finally average it and compute AUC
		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		plt.figure()
		lw = 2
		index = "macro"
		plt.plot(fpr[index], tpr[index], color='darkorange',
				lw=lw, label='ROC curve %s (area = %0.2f)' % (index, roc_auc[index]))
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.savefig("ROC_{}.png".format(self.name))
		
		with open("prob_train_{}.csv".format(self.name), "w") as fp_train, \
			open("prob_test_{}.csv".format(self.name), "w") as fp_test:
			title = "{:50s} {:4s}  {:5s}  ".format("annotation", "line", "label")
			title += "".join(["{:12s}".format("score_" + str(i)) for i in range(20)])
			title += "\n"
			# print(title)
			content_train = title
			content_test = title
			for name, label, data in zip(self.train_file.name, self.train_file.label, self.train_input):
				prob = ["{:12.10f}".format(p) for p in data]
				annotation = os.path.basename(os.path.dirname(name))
				line = os.path.basename(name)
				content_train += "{:50s} {:3s} {:7d} {} \n".format(annotation, line, label, " ".join(prob))
			fp_train.write(content_train)
			for name, label, data in zip(self.test_file.name, self.test_file.label, self.test_input):
				prob = ["{:9.7f}".format(p) for p in data]
				annotation = os.path.basename(os.path.dirname(name))
				line = os.path.basename(name)
				content_test += "{:50s} {:3s} {:7d} {} \n".format(annotation, line, label, " ".join(prob))
			fp_test.write(content_test)