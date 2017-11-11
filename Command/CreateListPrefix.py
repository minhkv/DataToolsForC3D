from __future__ import print_function
from Command import *
import os
import sys
import copy
sys.path.append("..")
sys.path.append("../Model")
from Video import *
import config

"""
    Create List Prefix for extracting feature or finetuning or testing
    - Finetuning: provide split_file, input_folder, use_image
    - Testing: 
    - Feature extraction: provide input_folder, use_image
"""
class CreateListPrefix(Command):
    def __init__(
        self, 
        split_file,
        output_feature_file=None, 
        use_image=True):
        self.split_file = split_file
        self.output_feature_file = output_feature_file
        self.use_image = use_image

    # def write_chunk_list_to_file(self, path, syntax, chunk_list):
    #     with open(path, "w") as fp:
    #         for args in chunk_list:
    #             # line = syntax.format(*chunk_list)
    #             fp.write(syntax.format(*args))
    # def create_chunk_list(self, name, label, num_frames):
    #     chunk_list = []
    #     start_frame = []
    #     start = 0
    #     if self.use_image:
    #         start = 1
    #     for num_frame in num_frames:
    #         start_frame.append(range(start, num_frame - 32, 16))
    #     for i, name_item in enumerate(name):
    #         for num_frame in start_frame[i]:
    #             chunk_list.append([name_item, num_frame, label[i]])
    #     return chunk_list
    def execute(self):
        print("[Info] Creating input_chunk_list")
        # input_chunk_list = self.create_chunk_list(
        #     self.split_file.name,
        #     self.split_file.label,
        #     self.split_file.num_frames
        # )
        # print ("[Info] Writing input_chunk_list to file")
        # self.write_chunk_list_to_file(os.path.join(config.temp, "input.txt"), "{0} {1} {2}\n", input_chunk_list)
        self.split_file.create_chunk_list()
        print ("[Info] Writing input_chunk_list to file")
        self.split_file.write_chunk_list_to_file()
        if self.output_feature_file != None: # if extract feature
            print("[Info] Creating output_chunk_list")
            # output_chunk_list = self.create_chunk_list(
            #     self.output_feature_file.name,
            #     self.output_feature_file.label,
            #     self.output_feature_file.num_frames
            # )
            self.output_feature_file.create_chunk_list()
            print ("[Info] Writing output_chunk_list to file")
            # self.write_chunk_list_to_file(os.path.join(config.temp, "output.txt"), "{0}/{1:06d}\n", output_chunk_list)        
            self.output_feature_file.write_chunk_list_to_file()