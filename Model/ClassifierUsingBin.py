from __future__ import print_function
from Classifier import *
import array
class ClassifierUsingBin(Classifier):
    def read_bin(self, path):
        f = open(path, "rb")
        # read all bytes into a string
        s = f.read()
        f.close()
        (n, c, l, h, w) = array.array("i", s[:20])
        feature_vec = np.array(array.array("f", s[20:]))
        return feature_vec
    def get_list_feature_in_folder(self, path, layer):
        listfiles = glob.glob(os.path.join(path, "*" + layer))
        return listfiles
    def load_feature_from_folder_and_average(self, split_file):
        split_input = []
        split_label = []
        count_loss = 0
        try:
            for bin_folder, label in zip(split_file.name, split_file.label):
                # print("[Info] Loading feature from: {}".format(os.path.basename(bin_folder)))
                list_feature = self.get_list_feature_in_folder(bin_folder, self.layer)
                if len(list_feature) == 0:
                    count_loss += 1
                    continue
                features = [self.read_bin(bin_file) for bin_file in list_feature]
                features = np.mean(features, axis=0)
                split_input.append(features)
                split_label.append(label)
            print("Lost: {}".format(count_loss))
        except IOError as er:
            print("[Error] Error: {}".format(er))
        return split_input, split_label
