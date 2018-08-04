from __future__ import print_function
from collections import OrderedDict
import os

annotation_path = "/home/minhkv/datasets/Kinect_vis_annotate/v5/Kinect3"
mapping_data = OrderedDict()

list_ann = [os.path.join(annotation_path, ann) for ann in os.listdir(annotation_path)]

