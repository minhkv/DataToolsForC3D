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
    - Finetuning: provide split_file, input_folder
    - Testing: 
    - Feature extraction: provide input_folders
"""
class CreateListPrefix(Command):
    def __init__(
        self, 
        split_file,
        output_feature_file=None, 
        ):
        self.split_file = split_file
        self.output_feature_file = output_feature_file

    def execute(self):
        self.split_file.create_chunk_list()
        self.split_file.write_chunk_list_to_file()
        if self.output_feature_file != None: # if extract feature
            self.output_feature_file.create_chunk_list()
            self.output_feature_file.write_chunk_list_to_file()