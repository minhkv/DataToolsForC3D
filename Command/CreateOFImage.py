from __future__ import print_function
from Command import *
from utils.create_of_image import *

class CreateOFImage(Command):
    def __init__(self, u_file, v_file, output_file):
        self.u_file = u_file
        self.v_file = v_file
        self.output_file = output_file
    def execute(self):
        for u_folder, v_folder, flow_folder in zip(self.u_file.name, self.v_file.name, self.output_file.name):
            flows = create_of_image(u_folder, v_folder)
            save_of_image(flows, flow_folder, syntax="{:06d}.png")
            # print (os.path.exists(flow_folder))
        