from __future__ import print_function
import os
from Command import *
class CreateFeatureFolder(Command):
    def __init__(self, split_file):
        self.split_file = split_file
    def make_dir_if_non_exist(self, path):
        if os.path.exists(path):
            print("[Warning] Folder {} exists. Not create".format(path))
        else:
            os.makedirs(path)
    def execute(self):
        print("[Info] Creating binary feature folder")
        for folder in self.split_file.name:
            self.make_dir_if_non_exist(folder)