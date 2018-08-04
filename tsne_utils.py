from time import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import config_c3d
from Model.UCFSplitFile import UCFSplitFile
from config.config_mica import *

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None, list_class_to_plot=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    class_label = set(y)
    if list_class_to_plot != None:
        class_label = list_class_to_plot
    num_class = len(class_label)
    item_index = range(X.shape[0])

    plt.figure()
    ax = plt.subplot(111)
    for o, label in enumerate(class_label):
        class_index = [i for i in item_index if y[i]==label]
        plt.scatter(X[class_index, 0], 
            X[class_index, 1], 
            label=label, 
            color=plt.cm.tab20(o / float(num_class)))

    plt.xticks([]), plt.yticks([])
    # ax.legend(loc='upper right')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.05), shadow=True, ncol=1)

    if title is not None:
        plt.title(title)
def convert_and_subtract(label):
    return str(int(label) - 1)

classInd = UCFSplitFile(
    r"(?P<label>.+) (?P<name>\w+)", 
    class_ind_path)
classInd.load_name_and_label()
classInd.preprocess_label(convert_and_subtract)
print(classInd.map_label_to_name)

list_class_to_plot_ucf101 = [
    'ApplyEyeMakeup',
    'ApplyLipstick',
    'Archery',
    'BodyWeightSquats',
    'BoxingPunchingBag',
    'CricketBowling',
    'Haircut',
    'HandstandWalking',
    'Lunges',
    'MilitaryParade',
    'Nunchucks',
    'PlayingCello',
    'PlayingDaf',
    'PullUps',
    'PushUps',
    'ShavingBeard',
    'Shotput',
    'SoccerJuggling',
    'ThrowDiscus',
    'WallPushups'
]
list_class_to_plot_hmdb51 = [
    'brush_hair',
    'cartwheel',
    # 'chew',
    'clap',
    'drink',
    'eat',
    # 'fencing',
    'flic_flac',
    'hit',
    # 'jump',
    'laugh',
    'sit',
    'stand',
    'turn'
    # 'walk',
    # 'wave'
]
list_class_to_plot_mica = [
    "walk",
    "run_slowly",
    "static_jump",
    "move_hand_and_leg",
    "left_hand_pick_up",
    "right_hand_pick_up",
    "stagger",
    "front_fall",
    "back_fall",
    "left_fall",
    "right_fall",
    "crawl",
    "sit_on_chair_then_stand_up",
    "move_chair",
    "sit_on_chair_then_fall_left",
    "sit_on_chair_then_fall_right",
    "sit_on_bed_and_stand_up",
    "lie_on_bed_and_sit_up",
    "lie_on_bed_and_fall_left",
    "lie_on_bed_and_fall_right"
]
list_hmdb51_20_difficult = [
    'brush_hair',
    # 'chew',
    'climb',
    # 'climb_stairs',
    'dive',
    # 'draw_sword',
    'dribble',
    # 'fall_floor',
    # 'fencing',
    'flic_flac',
    'golf',
    # 'handstand',
    # 'hit',
    'hug',
    'kiss',
    # 'pour',
    'pullup',
    # 'punch',
    'push'#,
    # 'pushup',
]
