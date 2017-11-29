from __future__ import print_function
from Command import *
class Train(Command):
    def __init__(self, c3d):
        self.c3d = c3d
    def execute(self):
        self.c3d.train()
        