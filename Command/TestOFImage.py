from __future__ import print_function
import os
from Command import *
from utils.test_of_image import *
from utils.create_of_image import *

class TestOFImage(Command):
    def __init__(self, u_file, v_file, output_file):
        self.u_file = u_file
        self.v_file = v_file
        self.output_file = output_file
    def execute(self):
        quality = 0
        for u_folder, v_folder, flow_folder in zip(self.u_file.name, self.v_file.name, self.output_file.name):
            print("[Info] Testing flow: {}".format(os.path.basename(u_folder)))
            q = test_list_of(u_folder, v_folder, flow_folder, type_flow="png", number_result=True)
            print ("[Info] Quality: {}".format(q))
            quality += q
        print("[Info] Quality of optical flow: {0}/{1}".format(quality, float(len(self.u_file.name))))
        