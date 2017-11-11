from __future__ import print_function
import re
from Video import *
class UCFSplitFile: 
    """"""
    def __init__(
        self, 
        syntax, 
        path, 
        clip_size=16, 
        chunk_list_syntax="{0} {1} {2}\n", # required
        chunk_list_file="unknown_chunk.txt", # required
        use_image=True):
        self.syntax = syntax
        self.path = path
        self.name = []
        self.label = []
        self.map_label_to_name = {}
        self.map_name_to_label = {}
        self.num_frames = []
        self.clip_size = clip_size
        self.chunk_list = []
        self.chunk_list_syntax = chunk_list_syntax
        self.chunk_list_file = chunk_list_file
        self.use_image = use_image
    def load_name_and_label(self):
        try:
            with open(self.path, 'r') as fp:
                content = fp.readlines()
                for line in content:
                    m = re.compile(self.syntax)
                    d = m.search(line).groupdict()
                    if d.has_key('name'):
                        self.name.append(d['name'])
                    if d.has_key('label'):
                        self.label.append(d['label'])  
                self.map_label_to_name = dict(zip(self.label, self.name))
                self.map_name_to_label = dict(zip(self.name, self.label))
        except IOError as ex:
            print("[Error] File Error: " + str(ex))
    def concatenate(self, split_file):
        "Concatenate the name, label list of two UCFSplitFile"
        self.name.extend(split_file.name)
        self.label.extend(split_file.label)
    def count_frame(self):
        """This function only work after defined exactly filename"""
        if self.use_image:
            print ("Not implemented yet")
        else:
            for vid in self.name:
                video = Video(vid)
                num_fr, fps = video.get_frame_count()
                self.num_frames.append(num_fr)
    def create_chunk_list(self):
        start_frame = []
        start = 0
        if self.use_image:
            start = 1
        for num_frame in self.num_frames:
            start_frame.append(range(start, num_frame - 32, self.clip_size))
        for i, name_item in enumerate(self.name):
            for num_frame in start_frame[i]:
                self.chunk_list.append([name_item, num_frame, self.label[i]])
        
    def write_chunk_list_to_file(self):
        with open(self.chunk_list_file, "w") as fp:
            for args in self.chunk_list:
                line = self.chunk_list_syntax.format(*args)
                fp.write(line)
    def preprocess_name(self, process):
        self.name = [process(n) for n in self.name]
    def preprocess_label(self, process):
        self.label = [process(lb) for lb in self.label]
    def convert_name_to_label(self, name):
        return self.map_name_to_label[name]
    def convert_label_to_name(self, label):
        return self.map_label_to_name[label]
