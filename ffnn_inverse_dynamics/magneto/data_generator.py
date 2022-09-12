import os

import tensorflow as tf
import numpy as np
import math
import pickle

from gn_inverse_dynamics.utils.myutils import * 
from gn_inverse_dynamics.utils import mymath as mymath
from ffnn_inverse_dynamics.magneto.trajectory_data_generator import MagnetoJointTrajDataGenerator

class PassThresholdParam():
    def __init__(self, init_traj_pass_threshold=0, traj_pass_threshold=0):
        self.init_traj_pass_threshold = init_traj_pass_threshold
        self.traj_pass_threshold = traj_pass_threshold

class FFDataSetGenerator():
    def __init__(self, traj_data_path, pass_param=None):
        log_with_time("FFDataSetGenerator")        
        self.data_size = None
        self.static_data = None

        # check traj_data_path, urdf_data_path
        traj_data_path = self.get_path_str(traj_data_path,'traj_data_path')

        self.trajectory_generator = MagnetoJointTrajDataGenerator(traj_data_path, pass_param) 

    def get_path_str(self, path, print_prefix=''):
        log_with_time(" graph generator- {}: {}".format(print_prefix, path))
        if(type(path)!=str):
            path = path.decode('utf-8')
        return path 
        

    def get_full_tfdataset(self):
        gen = self.gen_data_tsr()   
        data_signature = self.get_specs_from_data_tsr(next(gen))
        tfdataset = tf.data.Dataset.from_generator(self.gen_data_tsr,
                        output_signature = data_signature)
        return tfdataset, tfdataset.element_spec

    def get_element_spec(self):
        tfdataset, element_spec = self.get_full_tfdataset()
        return element_spec

    def get_data_size(self):
        return self.trajectory_generator.get_data_size()

    def gen_data_tsr(self):
        # yield input_data, target_data
        for input_data, target_data in self.trajectory_generator.gen_data_list():
            input_data_tsr = tf.convert_to_tensor(input_data)
            target_data_tsr = tf.convert_to_tensor(target_data)
            yield input_data_tsr, target_data_tsr

    def get_specs_from_data_tsr(self,data_tsr_tuples):
        return ( tf.TensorSpec.from_tensor(data_tsr_tuples[0]),
                tf.TensorSpec.from_tensor(data_tsr_tuples[1]) )
        
    # ======= TF DATASET GENERATION =======
    def generate_tf_dataset(self, 
                            dataset_path, dataset_size, 
                            args):
        log_with_time("GENERATE TFDATASET")

        ## DATASET PATHS        
        print("dataset_path = {}".format(dataset_path))          
        full_dataset_path = dataset_path + '/fulldataset'
        train_dataset_path = dataset_path + '/traindataset'
        val_dataset_path = dataset_path + '/validationdataset'
        test_dataset_path = dataset_path + '/testdataset'
        create_folder(dataset_path)
        delete_and_create_folder(full_dataset_path)
        delete_and_create_folder(train_dataset_path)
        delete_and_create_folder(val_dataset_path)
        delete_and_create_folder(test_dataset_path)


        ## GENERATE FULL TF DATASET
        full_dataset, dataset_element_spec  = self.get_full_tfdataset()

        # SAVE FULL DATA & DATA ELEMENT SPEC
        log_with_time("full data save start")
        tf.data.experimental.save(full_dataset, full_dataset_path)
        # # COUNT data_sdataset_sizeize
        if dataset_size == None:            
            if self.data_size == None :
                dataset_size = self.get_data_size() - 1
            else:
                dataset_size = self.data_size

        print("dataset_size={}".format(dataset_size))
        # SAVE element_spec/data_size
        with open(dataset_path + '/element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
            pickle.dump(dataset_element_spec, out_)

        with open(dataset_path + '/data_size', 'wb') as out_:  # also save the element_spec to disk for future loading
            pickle.dump(dataset_size, out_)

        ## LOAD FULL DATA 
        log_with_time("full data reload")
        with open(dataset_path + '/element_spec', 'rb') as in_:
            dataset_element_spec = pickle.load(in_)
        with open(dataset_path + '/data_size', 'rb') as in_:
            dataset_size = pickle.load(in_)
        full_dataset = tf.data.experimental.load(full_dataset_path, 
                                element_spec = dataset_element_spec)

        if(args.split_data==False):
            print("no need split_data")
            return

        if(args.shuffle):
            print("data is shuffled")
            full_dataset = full_dataset.shuffle(buffer_size = dataset_size)

        # SAVE SPLIT DATA
        # train_size = int(args.train_ratio * dataset_size)
        val_size = max( 64, int(args.val_ratio * dataset_size) )
        test_size = max( 64, int(args.test_ratio * dataset_size) )
        train_size = dataset_size - val_size - test_size
        log_with_time("dataset_size = train/val/test : {}/{}/{}".format(
                                train_size, val_size, test_size))

        train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)
        
        log_with_time("train data save start")
        tf.data.experimental.save(train_dataset, train_dataset_path)
        with open(dataset_path + '/train_data_size', 'wb') as out_:  
            pickle.dump(train_size, out_)

        log_with_time("validation data save start")
        tf.data.experimental.save(val_dataset, val_dataset_path)
        with open(dataset_path + '/val_data_size', 'wb') as out_:  
            pickle.dump(val_size, out_)

        log_with_time("test data save start")
        tf.data.experimental.save(test_dataset, test_dataset_path)
        with open(dataset_path + '/test_data_size', 'wb') as out_:  
            pickle.dump(test_size, out_)

