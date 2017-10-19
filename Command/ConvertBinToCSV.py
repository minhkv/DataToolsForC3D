from __future__ import print_function
import os
import glob
import array
import numpy as np
from Command import *
class ConvertBinToCSV(Command):
    def __init__(self, input_split_file, output_split_file, layer):
        self.input_split_file = input_split_file
        self.output_split_file = output_split_file
        self.layer = layer
    def get_list_feature_in_folder(self, path, layer):
        listfiles = glob.glob(os.path.join(path, "*" + layer))
        return listfiles
    def save_bin_feature_to_csv(self, feature, output_filename):
        np.savetxt(
            output_filename,
            feature[None, :],
            fmt='%.16f',
            delimiter=','
            )
    # Convert binary feature to csv file
    def get_features(self, feature_files, feature_layer):
        ''' From binary feature files, take an average (for multiple clips) '''

        # in case of a single feature_file, force it to a list
        if isinstance(feature_files, basestring):
            feature_files = [feature_files]

        # read each feature, take an an average
        for clip_count, feature_file in enumerate(feature_files):
            # print ("clip_count={}, feature_file={}".format(clip_count, feature_file))
            if not os.path.exists(feature_file):
                feature_file += '.' + feature_layer

            if not os.path.exists(feature_file):
                print ("[Error] feature_file={} does not exist!".format(feature_file))
                return None

            # read binary data
            f = open(feature_file, "rb")
            # read all bytes into a string
            s = f.read()
            f.close()
            (n, c, l, h, w) = array.array("i", s[:20])
            feature_vec = np.array(array.array("f", s[20:]))

            if clip_count == 0:
                feature_vec_avg = feature_vec
            else:
                feature_vec_avg += feature_vec
        feature_vec_avg = feature_vec_avg / len(feature_files)

        return feature_vec_avg
    
    def execute(self):
        print("[Info] Converting binary feature to csv:")
        for input_bin_folder, output_feature_folder in zip(self.input_split_file.name, self.output_split_file.name):
            print("[Info] Processing: {}".format(os.path.basename(input_bin_folder)))
            feature_files = self.get_list_feature_in_folder(input_bin_folder, self.layer)
            for feature_path in feature_files:
                feature_start_frame = os.path.splitext(os.path.basename(feature_path))[0]
                feature = self.get_features(feature_path, self.layer)
                output_csv = os.path.join(output_feature_folder, "{1}.csv".format(feature_start_frame))
                print(output_csv)
                self.save_bin_feature_to_csv(feature, output_csv))
        return