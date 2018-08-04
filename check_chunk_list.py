from __future__ import print_function
import os



tmp_path = "Asset/tmp"
input_file = os.path.join(tmp_path, "input.txt")
output_file = os.path.join(tmp_path, "output.txt")

with open(input_file) as ip, \
    open(output_file) as op:
    in_content = ip.readlines()
    out_content = op.readlines()
    i = 0
    for in_line, out_line in zip(in_content, out_content):
        print(i)
        i += 1
        in_numframe = int(in_line.split(' ')[1])
        out_numframe = int(os.path.basename(out_line))
        in_folder = os.path.basename(in_line.split(' ')[0])
        out_folder = os.path.basename(os.path.dirname(out_line))
        print("{}_{} {}_{}".format(in_numframe, in_folder, out_numframe, out_folder))
        if not (in_numframe == out_numframe and in_folder == out_folder):
            print("False")
            break