from __future__ import print_function
import matplotlib, os, sys, numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from config.config_hmdb51 import *

def read_confusion_matrix(path):
    cf = []
    with open(path, "r") as fp:
        content = fp.readlines()
        for line in content:
            row = np.array([int(n.strip()) for n in line.strip().split(',')])
            cf.append(row)
        cf = np.array(cf)
    return cf

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # for i in range(len(tick_marks)):
        # if i % 5 != 0:
            # tick_marks[i] = ""
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

name = sys.argv[1]
# title= sys.argv[2]
dataset = ""
cf_matrix_path = os.path.join("report", dataset,"report_{}".format(name), "confusion_matrix_{}.csv".format(name))
image = os.path.join("report", dataset, "report_{}".format(name), "confusion_matrix_{}.png".format(name))
jpg = os.path.join("report", dataset, "report_{}".format(name), "confusion_matrix_{}.jpg".format(name))
cf = read_confusion_matrix(cf_matrix_path)
plot_confusion_matrix(cf, normalize=True, classes=[])#, title='{}'.format(title))
# plt.imshow(cf)
plt.savefig(image, format='png', dpi=100)
# plt.imshow(cf)
# plt.savefig(jpg)