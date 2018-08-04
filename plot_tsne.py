from tsne_utils import *

t0 = time()
X_tsne = []
y = []
y_pred = []
report_folder = "{0}/report_{1}".format(config_c3d.report_folder, config_c3d.classifier_name)
#file_name = os.path.join(report_folder, 'train_tsne_{}.csv'.format(config_c3d.classifier_name))
#out_file_name = os.path.join(report_folder, "train_tsne_20_{}.png".format(config_c3d.classifier_name))
file_name = 'train_tsne_{}.csv'.format(config_c3d.classifier_name)
out_file_name = 'train_tsne_{}.png'.format(config_c3d.classifier_name)
file_name = os.path.join(report_folder, file_name)
out_file_name = os.path.join(report_folder, out_file_name)
with open(file_name, "r") as fp:
    content = fp.readlines()
    for line in content:
        item = line.strip().split(" ")
        X_tsne.append([float(item[1]), float(item[2])])
        l = str(int(item[0]) + 1)
        y.append(classInd.convert_label_to_name(l))
        # y.append(item[0])


plot_embedding(np.array(X_tsne), np.array(y),
               "CMDFALL Early Fusion",
               list_class_to_plot=list_class_to_plot_mica)

plt.savefig((out_file_name))
