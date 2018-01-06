from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
# from skimage.io import *
import glob
import cv2
import scipy.misc
from PIL import Image

def read_image(path, type_read):
    im = cv2.imread(path, type_read)
    return im

def write_image(fname, im):
    cv2.imwrite(fname, im)
    # cv2.imwrite(fname, im, [int(cv2.IMWRITE_JPEG_QUALITY), 110])
    return

def stack_image(u_path, v_path):
    flow_u = read_image(u_path, cv2.IMREAD_GRAYSCALE)
    flow_v = read_image(v_path, cv2.IMREAD_GRAYSCALE)
    mag = np.zeros_like(flow_u)
    flow = np.stack((flow_u, flow_v, mag), axis=-1)
    # flow = np.stack((flow_u, flow_v), axis=-1)
    return flow 
def list_image_in_folder(folder, type_image="jpg"):
    # list_image = glob.glob(os.path.join(folder, "*.{}".format(type_image)))
    # list_image.sort()
    list_image = sorted([os.path.join(folder, f) for f in os.listdir(folder) if type_image in f])
    return list_image
def create_of_image(u_folder, v_folder):
    list_flow_u = list_image_in_folder(u_folder)
    list_flow_v = list_image_in_folder(v_folder)
    flows = np.array([stack_image(u_path, v_path) for u_path, v_path in zip(list_flow_u, list_flow_v)])
    return flows
def save_of_image(flows, output_folder, syntax="{:06d}.png"):
    print("[Info] Processing: {}".format(os.path.basename(output_folder)))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, f in enumerate(flows):
        fname = os.path.join(output_folder, syntax.format(i + 1))
        write_image(fname, f)
