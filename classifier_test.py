from classifier_util2 import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random

# rgb <fc7, prob>: /home/minhkv/datasets/feature/csv/rgb/finetune/<fc7, prob>/UCF101
# optic <fc7, prob>: /home/minhkv/datasets/feature/csv/optic/<fc7, prob>/optic/
# optic_precalculate<fc7, prob>: /home/minhkv/datasets/feature/csv/optic_precalculate
#                                /home/minhkv/datasets/feature/csv/prob_precalculate/output

print 'Loading data'
csv_dir = '/home/minhkv/datasets_old/feature/csv/rgb/finetune/fc7/UCF101'
X_train, y_train, X_test, y_test = load_train_test_split(csv_dir)

s_index = range(len(y_train))
random.shuffle(s_index)
shuffle_data(X_train, s_index)
shuffle_data(y_train, s_index)

# /media/usb/export/minhkv/classifier/rgb/<X_train_rgb_1.pkl, X_test_rgb_1.pkl>

# X_train = load_variable('/media/usb/export/minhkv/classifier/rgb/X_train_rgb_1.pkl')
# y_train = load_variable('/media/usb/export/minhkv/classifier/y_train_optic_1.pkl')
# X_test = load_variable('/media/usb/export/minhkv/classifier/rgb/X_test_rgb_1.pkl')
# y_test = load_variable('/media/usb/export/minhkv/classifier/y_test_optic_1.pkl')

names = ["NearestNeighbors5", "LinearSVM", "RBFSVM", "GaussianProcess",
         "DecisionTree", "RandomForest", "MLPClassifier", "AdaBoost",
         "NaiveBayes", "QuadraticDiscriminantAnalysis"]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
index = 1
print '.....................Training .....................'
clf = create_classifier(classifiers[index], names[index], X_train, y_train)
# classifier/clf_optic_t1_fc7.pkl
# Optic: /media/usb/export/minhkv/classifier/<LinearSVM_optic_t1_fc7.pkl, RBFSVM_optic_t1_fc7.pkl>
# RGB: /media/usb/export/minhkv/classifier/rgb/ <>
#clf = load_variable('/media/usb/export/minhkv/classifier/LinearSVM_optic_t1_fc7.pkl')

print '.....................Testing ......................'
y_pred, pre, recall, acc = get_score(clf, X_test, y_test)



# cf_matrix = confusion_matrix(y_test, y_pred, labels=label)
#save_variable(cf_matrix, '/media/usb/export/minhkv/classifier/cf_optic' + names[index] + '.pkl')

print '....................Demo.............................'
print 'Loading ~/datasets/feature/csv/optic_precalculate/v_ApplyEyeMakeup_g01_c01/v_ApplyEyeMakeup_g01_c01.csv'





    
