from __future__ import print_function
import cv2
import sys
class Video:
    def __init__(self, path):
        self.path = path
    def get_frame_count(self):
        ''' Get frame counts and FPS for a video '''
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print ("[Error] video={} can not be opened.".format(self.path))
            sys.exit(-6)

        # get frame counts
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # in case, fps was not available, use default of 29.97
        if not fps or fps != fps:
            fps = 29.97

        return num_frames, fps

    # def extract_frames():
# vid = Video("/home/minhkv/datasets/UCF101/v_ApplyEyeMakeup_g08_c01.avi")
# num, fps = vid.get_frame_count()
# print (num)
# print (fps)