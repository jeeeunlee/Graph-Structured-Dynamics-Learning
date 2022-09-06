from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time
import os 
import sys
import pickle

# CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

import numpy as np
import tensorflow as tf
import sonnet as snt 

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf

from typical_gnn_inverse_dynamics.my_graph_nets import gn_models as myModels

from typical_gnn_inverse_dynamics.utils.mygraphutils import *
from typical_gnn_inverse_dynamics.utils.myutils import *

from typical_gnn_inverse_dynamics.train_model.run_functions import *

class TrainModel():
    def __init__(self, args, name="TrainModel"):
        self.name = name
        self._set_logger(args)        
        self._set_params(args)
        self._set_dataset(args) # dataset, valdataset
        self._set_save_and_load_model(args) # model /save,load path
        self._set_optimizer()
        self._log_train_info(args)

    def get_dataset_signature(self, dataset):
        # return get_batch_specs_from_graph_tuples_batch_sets(dataset)
        return get_specs_from_graph_tuples_batch_sets(dataset) #reshaped

    def update_dataset_batch(self):
        if(self.buffer_size):
            self.dataset_batch = self.dataset.shuffle(buffer_size = self.buffer_size
                                ).batch(self.batch_size, drop_remainder=True)
            self.dataset_batch_signature = self.get_dataset_signature(self.dataset_batch)

    def save_model(self, batch_loss_sum):
        if( self.min_loss > batch_loss_sum and 
                self.SAVE_MODEL_TIMER.check(self.save_every_seconds) ):
            print(" model saving start " )
            self.min_loss = batch_loss_sum
            to_save = snt.Module()
            to_save.all_variables = list(self.model.variables) 
            tf.saved_model.save(to_save, self.save_model_path)
            print(" model saved end " )

    def save_sample_input(self, input_tr, file_str = "input_tr"):
        with open(self.save_model_path + '/' + file_str, 'wb') as out_:  
            # also save the input_tr to disk for future loading
            pickle.dump(input_tr, out_)

    def create_loss_ops(self, target_op, output_ops):
        ''' Create supervised loss operations from targets and outputs.
        Args:
            target_op: The tensor of target torque (edge).
            output_ops: The list of output graphs from the model.

        Returns:
            A list of loss values (tf.Tensor), one per output op.
        '''
        loss_ops = [tf.reduce_mean(
                    tf.reduce_mean((output_op.edges - target_op.edges)**2, axis=-1))
                    for output_op in output_ops   ]
        return loss_ops

    def print_loss_ops(self, target_op, output_ops):
        ''' Create supervised loss operations from targets and outputs.
        Args:
            target_op: The tensor of target torque (edge).
            output_ops: The list of output graphs from the model.

        Returns:
            A list of loss values (tf.Tensor), one per output op.
        '''
        loss_ops=list()
        for output_op in output_ops:
            loss1 = tf.reduce_mean((output_op.edges - target_op.edges)**2, axis=-1)
            loss2 = tf.reduce_mean(loss1)
            #print(loss2)
            loss_ops.append( loss2 )
        return loss_ops
    
    # ======================================
    #       Initial setting functions
    # ======================================        
    def _set_logger(self, args):
        # Make log Directory
        self.save_every_seconds = 300 # = 5min
        if(args.log_dir == ''):
            log_dir = os.path.join("a_result", get_local_time() )
        else:
            log_dir = os.path.join("a_result", args.log_dir)
        log_dir_path = os.path.join(CURRENT_DIR_PATH, log_dir)
        create_folder(log_dir_path)

        self.log_dir_path = log_dir_path
        self.logf = Logger(log_dir_path + '/time_per_loss.csv')
        self.logf_val = Logger(log_dir_path + '/time_per_loss_val.csv')
        self.logf_test = Logger(log_dir_path + '/time_per_loss_test.csv')
        self.logf_info = Logger(log_dir_path + '/info.csv')

        self.TOTAL_TIMER = Timer()
        self.SAVE_MODEL_TIMER = Timer()

    def _log_train_info(self, args):
        self.logf_info.record_string("train model = {}".format(self.name))
        self.logf_info.record_string("dataset = {}".format(self.dataset_path))
        self.logf_info.record_string("model_save_path = {}".format(self.save_model_path))
        self.logf_info.record_string("edge_output_size no = {}".format(self.edge_output_size))
        self.logf_info.record_string("latent no = {}".format(self.edge_latent_size))
        self.logf_info.record_string("num_processing_steps = {}".format(self.num_processing_steps))
        self.logf_info.record_string("batch size = {}".format(self.batch_size))
        self.logf_info.record_string("learning rate = {}".format(self.learning_rate ))
        self.logf_info.record_string("validation_test = {}".format(self.validation_test ))
        

    def _set_params(self,args):
        self.learning_rate = args.learning_rate #5e-4
        self.num_processing_steps = args.num_processing_steps
        self.edge_latent_size = args.edge_latent_size
        self.batch_size = args.batch_size
        self.epoch_size = args.epoch_size
        self.buffer_size = args.buffer_size        
        self.validation_test = args.validation_test

        self.edge_output_size = args.edge_output_size


    def _set_dataset(self, args):
        self.dataset_path = os.path.join( CURRENT_DIR_PATH, args.dataset_path )

        with open(self.dataset_path + '/element_spec', 'rb') as in_:
            dataset_element_spec = pickle.load(in_)
        
        if(self.buffer_size):
            with open(self.dataset_path + '/data_size', 'rb') as in_:
                dataset_size = pickle.load(in_)
                self.buffer_size = min(dataset_size, self.buffer_size)
        

        # load train dataset, validation dataset
        self.dataset = get_dataset(self.dataset_path, dataset_element_spec, "train")
        # generate batch data
        self.dataset_batch = self.dataset.batch(self.batch_size, drop_remainder=True)        
        self.dataset_batch_signature = self.get_dataset_signature(self.dataset_batch)

        if(self.validation_test):
            self.valdataset = get_dataset(self.dataset_path, dataset_element_spec, "validation")
            self.valdataset_batch = self.valdataset.batch(self.batch_size, drop_remainder=True)
        else:
            self.valdataset = None
            self.valdataset_batch = None
        
    def _set_save_and_load_model(self, args):
        # save model parameter Directory : a_result/xxxx(time)/saved_model        
        self.save_model_path = self.log_dir_path + args.save_model_path 
        print("saved model path  : {} ".format(self.log_dir_path))
        print("saved model path  : {} ".format(self.save_model_path))
        create_folder(self.save_model_path) 
        
        ## define gn_model
        self.model = myModels.EncodeProcessDecode(edge_latent_size=self.edge_latent_size, 
                                                  edge_output_size=self.edge_output_size)
        ## initialize gn_model
        for train_traj_tr in self.dataset_batch :
            inputs_tr = graph_reshape(train_traj_tr[0])
            #targets_tr = graph_reshape(train_traj_tr[1])
            break
        _ = self.model(inputs_tr, self.num_processing_steps)

        # self.dataset_batch_signature = self.get_dataset_signature(self.dataset_batch)

        ## load model
        if(args.load_model == True):
            load_model_path = os.path.join( CURRENT_DIR_PATH, args.load_model_path)
            check_model(self.model)            
            load_model(self.model, load_model_path )
            check_model(self.model)

    def _set_optimizer(self):
        self.optimizer = snt.optimizers.Adam(self.learning_rate) 

