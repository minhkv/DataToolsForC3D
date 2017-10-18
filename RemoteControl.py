from __future__ import print_function

class RemoteControl(list):
    def __init__(self):
        list.__init__([])
    def run(self, slot):
        self[slot].execute()
        
    