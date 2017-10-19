from __future__ import print_function
from Command import *
class DrawLearningCurve(Command):
    def __init__(self, c3d):
        self.c3d = c3d
    def execute(self):
        self.c3d.draw_learning_curve()