from __future__ import print_function
import os
import sys
import subprocess
class C3D: 
    def __init__(
        self, 
        root_folder, 
        input_prefix, 
        output_prefix=None, 
        solver_config=None, 
        mean_file=None, 
        pre_trained=None, 
        training=True,
        use_image=True):
        asset_path = os.path.abspath('Asset')
        self.out_prototxt = os.path.abspath("Asset/tmp")

        self.root_folder = root_folder
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        if training:
            self.model_config = os.path.join(asset_path, 'c3d_ucf101_finetuning_train.prototxt')
        else:
            self.model_config = os.path.join(asset_path, 'c3d_ucf101_finetuning_test.prototxt')

        mean_file = os.path.join(self.out_prototxt, 'mean.binaryproto')
        if mean_file != None:
            self.mean_file = mean_file
            
        self.pre_trained = os.path.join(asset_path, "pretrained")
        if self.pre_trained != None:
            self.pre_trained = pre_trained
        
        self.solver_config = os.path.join(asset_path, "c3d_ucf101_finetuning_solver.prototxt")
        if solver_config != None:
            self.solver_config = solver_config
        
        self.use_image = use_image
    
    def generate_config_file(self, in_file, out_file, arg_list): 
        """ Format string store in in_file to generate out_file """
        try:
            with open(in_file, 'r') as fp:
                content = fp.readlines()
                content = ''.join(content)
                content = content.format(*arg_list)
                model_prototxt = os.path.join(self.out_prototxt, out_file)
                with open(model_prototxt, 'w') as mp:
                    print(content, file=mp)
        except IOError as ex:
            print ("IOError: " + str(ex))
    def generate_prototxt(self):
        model_out = "model.prototxt"
        solver_out = "solver.prototxt"
        use_image = "false"
        if self.use_image:
            use_image = "true"
        self.generate_config_file(self.model_config, model_out, [self.input_prefix, self.mean_file, use_image])
        self.generate_config_file(self.solver_config, solver_out, [os.path.join(self.out_prototxt, model_out), ""])
        
    def compute_volume_mean(self):
        tools = os.path.join(self.root_folder, "build", "tools")
        compute_volume_mean_bin = os.path.join(tools, "compute_volume_mean_from_list_videos.bin")
        if self.use_image:
            compute_volume_mean_bin = os.path.join(tools, "compute_volume_mean_from_list.bin")
        length = 16
        height = 128
        width = 171
        sampling_rate = 1
        output_file = os.path.join(self.out_prototxt, "mean.binaryproto")
        dropping_rate = 1
        cmd = [
            "GLOG_logtostderr=1",
            compute_volume_mean_bin,
            self.input_prefix,
            str(length),
            str(height),
            str(width),
			str(sampling_rate),
			output_file,
            str(dropping_rate)
        ]
        print("[Info] compute_volume_mean: {}".format(' '.join(cmd)))
        
        return_code = os.system(' '.join(cmd))
        print("Return code: {}".format(return_code))
        return return_code        
        
    #def feature_extraction(self):
        
        
    def finetune(self):
        fine_tune_bin = os.path.join(self.root_folder, "build", "tools", "finetune_net.bin")
        
        solver = os.path.join(self.out_prototxt, "solver.prototxt")
        cmd = [
            "GLOG_logtostderr=1",
            fine_tune_bin,
            solver,
            self.pre_trained
        ]
        print("[Info] Finetune: {}".format(' '.join(cmd)))
        return_code = os.system(' '.join(cmd))
        return return_code

    def count_line(self, path):
        try:
            with open(path) as fp:
                content = fp.readlines()
                return len(content)
        except:
            print("[Error] Cannot read file: {}".format(path))
            sys.exit(-6)
    def test_net(self):
        """Not available yet"""
        batch_size = 30
        num_batch = self.count_line(self.input_prefix) / batch_size
        gpu_id = 0
        test_net_bin = os.path.join(self.root_folder, "build/tools/test_net.bin")
        
        model = os.path.join(self.out_prototxt, "model.prototxt")
        cmd = [
            "GLOG_logtostderr=1",
            test_net_bin,
            model,
            self.pre_trained,
            str(num_batch),
            "GPU",
            str(gpu_id)
        ]
        print("[Info] Test net: {}".format(' '.join(cmd)))
        return_code = os.system(' '.join(cmd))
        return return_code

