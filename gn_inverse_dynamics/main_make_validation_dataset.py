import collections
import itertools
import os 
import sys

import tensorflow as tf
import numpy as np
import math
import pickle

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.utils import myutils

# load full dataset
# shuffle
# divide 
# 256, 512, 1024, 2056, 

class DataDivider():
    def __init__(self, args):
        self.args = args

    def run(self):
        load_dataset_path = CURRENT_DIR_PATH + self.args.load_dataset_path
        save_dataset_path = CURRENT_DIR_PATH + self.args.save_dataset_path

        print("load_dataset_path = {}".format(load_dataset_path))    
        print("save_dataset_path = {}".format(save_dataset_path))    

        # load_dataset_path = 
        full_dataset_path = load_dataset_path + '/fulldataset'
        train_dataset_path_1 = save_dataset_path + '/traindataset_1'
        train_dataset_path_2 = save_dataset_path + '/traindataset_2'
        train_dataset_path_3 = save_dataset_path + '/traindataset_3'
        train_dataset_path_4 = save_dataset_path + '/traindataset_4'
        train_dataset_path_5 = save_dataset_path + '/traindataset_5'
        val_dataset_path = save_dataset_path + '/validationdataset'

        myutils.create_folder(save_dataset_path)
        myutils.delete_and_create_folder(train_dataset_path_1)
        myutils.delete_and_create_folder(train_dataset_path_2)
        myutils.delete_and_create_folder(train_dataset_path_3)
        myutils.delete_and_create_folder(train_dataset_path_4)
        myutils.delete_and_create_folder(train_dataset_path_5)
        myutils.delete_and_create_folder(val_dataset_path)

        # load full dataset
        
        with open(load_dataset_path + '/element_spec', 'rb') as in_:
            dataset_element_spec = pickle.load(in_)
        with open(load_dataset_path + '/data_size', 'rb') as in_:
            dataset_size = pickle.load(in_)

        print("dataset_size = ")
        print(dataset_size)

        full_dataset = tf.data.experimental.load(full_dataset_path, 
                                element_spec = dataset_element_spec)

        # shuffle
        full_dataset = full_dataset.shuffle(buffer_size = dataset_size)
        
        # divide
        val_size = 256
        train_size = 256

        val_dataset = full_dataset.take(val_size)
        train_dataset = full_dataset.skip(val_size)

        train_dataset1 = train_dataset.take(train_size)
        train_dataset2 = train_dataset.take(train_size*2)
        train_dataset3 = train_dataset.take(train_size*4)
        train_dataset4 = train_dataset.take(train_size*8)
        train_dataset5 = train_dataset.take(train_size*16)        
        
        myutils.log_with_time("data size save start")      
        with open(save_dataset_path + '/val_data_size', 'wb') as out_:  
            pickle.dump(val_size, out_)        
        with open(save_dataset_path + '/train_data_size', 'wb') as out_:  
            pickle.dump(train_size, out_)
        with open(val_dataset_path + '/element_spec', 'wb') as out_:  
            pickle.dump(dataset_element_spec, out_)
        with open(train_dataset_path_1 + '/element_spec', 'wb') as out_:  
            pickle.dump(dataset_element_spec, out_)
        with open(train_dataset_path_2 + '/element_spec', 'wb') as out_:  
            pickle.dump(dataset_element_spec, out_)
        with open(train_dataset_path_3 + '/element_spec', 'wb') as out_:  
            pickle.dump(dataset_element_spec, out_)
        with open(train_dataset_path_4 + '/element_spec', 'wb') as out_:  
            pickle.dump(dataset_element_spec, out_)
        with open(train_dataset_path_5 + '/element_spec', 'wb') as out_:  
            pickle.dump(dataset_element_spec, out_)

        myutils.log_with_time("validation data save start")
        tf.data.experimental.save(val_dataset, val_dataset_path)

        myutils.log_with_time("train1 data save start")      
        tf.data.experimental.save(train_dataset1, train_dataset_path_1)

        myutils.log_with_time("train2 data save start")      
        tf.data.experimental.save(train_dataset2, train_dataset_path_2)

        myutils.log_with_time("train3 data save start")      
        tf.data.experimental.save(train_dataset3, train_dataset_path_3)

        myutils.log_with_time("train4 data save start")      
        tf.data.experimental.save(train_dataset4, train_dataset_path_4)

        myutils.log_with_time("train5 data save start")      
        tf.data.experimental.save(train_dataset5, train_dataset_path_5)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_dataset_path", type=str, default="/dataset/magneto/tfData_icra/case5")
    parser.add_argument("--save_dataset_path", type=str, default="/dataset/magneto/tfData_l4dc_validation_256")

    args = parser.parse_args()

    # SET DATA PATH
    data_divider = DataDivider(args)
    try:
        data_divider.run()
    except ValueError:
        exit()