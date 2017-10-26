from __future__ import print_function
import numpy as np
from Classifier import *
class ClassifierUsingProb(Classifier):
    def training(self):
        return
    def predict(self, x_test):
        return np.argmax(x_test)
    def testing(self): 
        self.test_pred = [self.predict(x_test) for x_test in self.test_input]
        self.confusion_matrix = confusion_matrix(
            y_true=self.test_label,
            y_pred=self.test_pred, 
            labels=range(len(self.class_ind.label))
            )