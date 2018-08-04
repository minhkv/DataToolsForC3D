from __future__ import print_function
import re
import os
import glob
from SplitFile import *
class MICASplitFile(SplitFile): 
    """"""
    def __init__(
        self, 
        syntax, 
        path, # list of file
        clip_size=16, 
        chunk_list_syntax="{0} {1} {2}\n", # required
        chunk_list_file="unknown_chunk.txt", # required
        use_image=True,
        type_image="png"):
        self.syntax = syntax
        self.path = path
        self.name = []
        self.segment_order = [] # concatenate
        self.start_in_video = [] # concatenate
        self.end_in_video = [] # concatenate
        self.label = [] # concatenate
        self.map_label_to_name = {}
        self.map_name_to_label = {}
        self.num_frames = [] # concatenate
        self.clip_size = clip_size
        self.chunk_list = []
        self.chunk_list_syntax = chunk_list_syntax
        self.chunk_list_file = chunk_list_file
        self.use_image = use_image
        self.type_image = type_image
    def load_name_and_label(self):
        try:
            print("[Load annotation] {}".format(self.path))
            with open(os.path.join(self.path), 'r') as fp:
                content = fp.readlines()
                for i, line in enumerate(content):
                    if len(line.strip()) == 0:
                        continue
                    m = re.compile(self.syntax)
                    d = m.search(line.strip()).groupdict()
                    if d.has_key('name'):
                        self.name.append(d['name'])
                    else:
                        self.name.append(self.path)
                    if d.has_key('order'):
                        self.segment_order.append(d['order'])
                    if d.has_key('label'):
                        self.label.append(d['label'])  
                    if d.has_key('start_in_video'):
                        self.start_in_video.append(int(d['start_in_video']))  
                    if d.has_key('end_in_video'):
                        self.end_in_video.append(int(d['end_in_video']))  
                self.map_label_to_name = dict(zip(self.label, self.name))
                self.map_name_to_label = dict(zip(self.name, self.label))
        except IOError as ex:
            print("[Error] File Error: " + str(ex))
    def concatenate(self, split_file):
        "Concatenate the name, label list of two UCFSplitFile"
        self.name.extend(split_file.name)
        self.label.extend(split_file.label)
        self.num_frames.extend(split_file.num_frames)
        self.start_in_video.extend(split_file.start_in_video)
        self.end_in_video.extend(split_file.end_in_video)
        self.segment_order.extend(split_file.segment_order)

    def count_frame(self):
        if self.use_image:
            for vid in self.name:
                
                list_image = [im for im in os.listdir(vid)]
                num_fr = len(list_image)
                self.num_frames.append(num_fr)
                print('[Count frame] {} {}'.format(vid, num_fr))
        else:
            for vid, start, end in zip(self.name, self.start_in_video, self.end_in_video):
                print('[Count frame] {}'.format(vid))
                num_fr = end - start + 1
                self.num_frames.append(num_fr)
    def create_chunk_list(self):
        print("[Info] Create chunk list")
        start_frame = []
        for start, end, num_frame in zip(self.start_in_video, self.end_in_video, self.num_frames):
            start_f = start
            if start == 0:
                start_f = 1
            start_frame.append(range(int(start_f), int(start_f) + num_frame - self.clip_size, self.clip_size))
        for i, name_item in enumerate(self.name):
            for num_frame in start_frame[i]:
                self.chunk_list.append([name_item, num_frame, self.label[i]])
