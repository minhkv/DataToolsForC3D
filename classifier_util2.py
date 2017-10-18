# Create a map number => class
# Read file csv in class folder
# Each csv file save in a vector then append to data 
# Create a output vector
# Train test split
# Training
# Testing
import os
import csv
import numpy as np
import glob
from sklearn import svm
from sklearn.model_selection import train_test_split
import cPickle as pickle
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from read_feature_file import *
from save_csv import *
import re
from sklearn.metrics import precision_score, recall_score, accuracy_score
import random
# import tqdm
# line syntax: <label> <class>
map_file = '/home/minhkv/datasets_old/data/ucfTrainTestlist/classInd.txt'
null_feature = '/home/minhkv/datasets_old/data/ucfTrainTestlist/NullFeature.txt'
with open(map_file) as f:
    content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    map_item = [x.strip().split(' ') for x in content]
    classname = [x.strip().split(' ')[1].lower() for x in content]
    label = [x.strip().split(' ')[0] for x in content]
    class_label = dict(zip(classname, label))
    label_class = dict(zip(label, classname))
    

def read_csv(filename):
    with open(filename, 'rb') as csvfile:
        feature=np.array([float(w) for w in csvfile.read().split(',')])
    csvfile.close()
    return feature



	
def load_data(list_file, bin=False):
    X_train = []
    y_train = []
    for csv_file in list_file:
        X_train.append(read_csv(csv_file[0]))
        y_train.append(csv_file[1])
    return (X_train, y_train)

def validate(X_test, y_test):
	
	y_test_predict = np.array(X_test).argmax(axis = 1)
	accurate = 0
	
	for i in range(len(y_test_predict)):
		y_test_predict[i] += 1
		y_test[i] = int(y_test[i])
	for i in range(len(y_test)):
		if (int(y_test_predict[i]) == int(y_test[i])):
			accurate += 1
	print accurate
	#print 'Accuracy ....'
	accuracy = float(accurate) / len(y_test_predict)
	print 'Accuracy: {}'.format(accuracy)
    
def save_variable(variable, path):
    with open(path, 'wb') as file:
        pickle.dump(variable, file)
        
def load_variable(path):
    with open(path, 'rb') as file:
        variable = pickle.load(file)
    return variable

def add_two_matrix(score_rgb, score_of, w=0.5):
    score_fusion = []
    if(w < 0 or w > 1):
        print 'w must between 0 and 1'
        w = 0.5
    for i in range(len(score_rgb)):
        row = []
        for j in range(len(score_rgb[i])):
            row.append(w * score_rgb[i][j] + (1-w) * score_of[i][j])
        score_fusion.append(row)
    return score_fusion

# line syntax: <class>/v_<class>_<group>_<chapter>.avi <label>

#m = re.match(r"(?P<folder>\w+)/v_(?P<class>\w+)_(?P<group>\w+)_(?P<chapter>\w+).avi (?P<label>\w+)", "YoYo/v_YoYo_g23_c06.avi 101")
#>>> m.groupdict()
#{'chapter': 'c06', 'folder': 'YoYo', 'group': 'g23', 'class': 'YoYo', 'label': '101'}

train_list_file = '/home/minhkv/datasets_old/data/ucfTrainTestlist/trainlist03.txt'
with open(train_list_file) as f:
    train_list = []
    train_label = []
    content = f.readlines()
    for line in content:
      m = re.match(r"(?P<folder>\w+)/v_(?P<class>\w+)_(?P<group>\w+)_(?P<chapter>\w+).avi (?P<label>\w+)", line)
      d = m.groupdict()
      train_list.append('_'.join(['v', d['class'], d['group'], d['chapter']]))
      train_label.append(d['label'])

# line syntax: <class>/v_<class>_<group>_<chapter>.avi 
test_list_file = '/home/minhkv/datasets_old/data/ucfTrainTestlist/testlist03.txt'
with open(test_list_file) as f:
    content = f.readlines()
    test_list = []
    test_label = []
    for x in content:
        test_file = x.strip().split(' ')[0].split('/')[1].split('.')[0]
        test_list.append(test_file)
        test_label.append(class_label[test_file.split('_')[1].lower()])

#syntax: data_list: v_class_group_chapter
#        label_list: list of integer
def build_list_file(csv_dir, data_list, label_list):
    list_file = []
    lost = 0
    for i in range(len(data_list)):
        data_class = 'v_' + map_item[(int)(label_list[i]) - 1][1]
        
        
        csv_file2 = os.path.join(csv_dir, data_list[i], data_list[i] + '.csv')
        csv_file1 = os.path.join(csv_dir, data_class, data_list[i] + '.csv')
        csv_file = os.path.join(csv_dir, data_class, data_list[i], data_list[i] + '.csv')
        
        if os.path.isfile(csv_file2):
            list_file.append([csv_file2, label_list[i]])
        elif os.path.isfile(csv_file):
            list_file.append([csv_file, label_list[i]])
        elif os.path.isfile(csv_file1):
            list_file.append([csv_file1, label_list[i]])
        else:
            list_file.append([null_feature, label_list[i]])
            lost +=1
    print ('Lost {:5d} files.'.format(lost))
    return list_file

def load_train_test_split(csv_dir):
    print ('Reading train data')
    # Load train data
    train_data = build_list_file(csv_dir, train_list, train_label)
    print(len(train_data))
    X_train, y_train = load_data(train_data)
    print ('Reading test data')
    # Load test data
    test_data = build_list_file(csv_dir, test_list, test_label)
    print len(test_data)
    X_test, y_test = load_data(test_data)
    return (X_train, y_train, X_test, y_test)

def shuffle_data(data, shuffle_index=None):
    if shuffle_index == None:
        shuffle_index = range(len(data))
        random.shuffle(shuffle_index)
    if not len(data) == len(shuffle_index):
        print '[WARNING] Data and index not have the same length'
    new_data = [data[s_index] for s_index in shuffle_index]
    return new_data

    

def compute_confusion_matrix(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    return cf_matrix

def create_classifier(clf, name, X_train, y_train):
    clf.fit(X_train, y_train)
    #save_variable(clf, 'classifier/' + name + '_optic_t1_fc7.pkl')
    return clf

def get_score(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    pre = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    return (y_pred, pre, recall, acc)

def normalize_vector(vec):
    norm1 = np.linalg.norm(vec) 
    if(norm1 == 0): 
        return vec
    else:
      return vec / norm1
    
def normalize_data(matrix, axis=0):
    count = 0
    for i in range(len(matrix)):
        if(np.linalg.norm(matrix[i]) == 0):
            count += 1
        matrix[i] = normalize_vector(matrix[i])
    # print 'Lost {} file '.format(count)
    return matrix

        
    
