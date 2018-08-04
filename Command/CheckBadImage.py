from __future__ import print_function
from Command import *
import numpy as np
import os


class CheckBadImage(Command):
    def __init__(self, split_file):
        self.split_file = split_file
    def execute(self):
        list_err_folder = []
        for name, start, end in zip(self.split_file.name, self.split_file.start_in_video, self.split_file.end_in_video):
            list_image = sorted(os.listdir(name))
            current_i = 0
            for i in range(start, end):
                # prev_im = "image_{:04d}.png".format(i - 1)
                prev_im = list_image[0]
                im = "image_{:04d}.png".format(i)
                
                contain = False
                for image_name in list_image:
                    if im in image_name:
                        contain = True
                        break
                if not contain:
                    list_err_folder.append(os.path.join(name, im))
                    cmd = [
                        "cp",
                        prev_im, 
                        os.path.join(name, im)
                    ]
                    cp_cmd = [
                        "cp",
                        os.path.join(name, prev_im), 
                        os.path.join(name, im)
                    ]
                    if not os.path.exists(os.path.join(name, prev_im)):
                        print(os.path.join(name, prev_im))
                    print(' '.join(cmd))
                    # code = os.system(" ".join(cp_cmd))
        with open("empty_depth.txt", "w") as fp:
            fp.write('\n'.join(list_err_folder))
            