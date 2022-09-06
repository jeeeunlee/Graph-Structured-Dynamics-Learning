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


from sn_inverse_dynamics import sn_model as myModels

from gn_inverse_dynamics.utils.myutils import *
from sn_inverse_dynamics.run_functions import *

# ============ ============ ============ ============ ============

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
        for data_batch in dataset:
            return ( tf.TensorSpec.from_tensor(data_batch[0]),
                     tf.TensorSpec.from_tensor(data_batch[1]) )    

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

    def create_loss_ops(self, target_op, output_op):
        ''' Create supervised loss operations from targets and outputs.
        Args:
            target_op: The tensor of target torque 
            output_op: The output from the model.

        Returns:
            loss values (tf.Tensor)
        '''        
        loss_ops = tf.reduce_mean(
                    tf.reduce_mean(
                        (output_op - target_op)**2, axis=-1))
        
        return loss_ops

    def print_loss_ops(self, target_op, output_op):
        ''' Create supervised loss operations from targets and outputs.
        Args:
            target_op: The tensor of target torque 
            output_op: The output from the model.

        Returns:
            loss values (tf.Tensor)
        '''
      
        loss1 = tf.reduce_mean((output_op - target_op)**2, axis=-1)
        loss2 = tf.reduce_mean(loss1)
        print(loss1)
        print(loss2)
        print("=================")
        print(tf.TensorSpec.from_tensor(target_op))
        print(tf.TensorSpec.from_tensor(output_op))
        print(tf.TensorSpec.from_tensor(loss1))
        print(tf.TensorSpec.from_tensor(loss2))
        print("=================")
        return loss2
    
    # ======================================
    #       Initial setting functions
    # ======================================        
    def _set_logger(self, args):
        # Make log Directory
        self.save_every_seconds = 30 # = 5min
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
        self.logf_info.record_string("output_size no = {}".format(self.output_size))
        self.logf_info.record_string("layer no = {}".format(self.num_layer))
        self.logf_info.record_string("latent no = {}".format(self.latent_size))
        self.logf_info.record_string("num_processing_steps = {}".format(self.num_processing_steps))
        self.logf_info.record_string("batch size = {}".format(self.batch_size))
        self.logf_info.record_string("learning rate = {}".format(self.learning_rate ))
        self.logf_info.record_string("validation_test = {}".format(self.validation_test ))
        

    def _set_params(self,args):
        self.learning_rate = args.learning_rate #5e-4
        self.latent_size = args.latent_size
        self.num_processing_steps=args.num_processing_steps
        self.batch_size = args.batch_size
        self.epoch_size = args.epoch_size
        self.buffer_size = args.buffer_size        
        self.validation_test = args.validation_test

        self.output_size = args.output_size
        self.num_layer = args.num_layer


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
        self.model = myModels.EncodeProcessDecode(latent_size=self.latent_size, 
                                                  output_size=self.output_size,
                                                  num_layer=self.num_layer,
                                                  num_processing_steps=self.num_processing_steps)
        ## initialize gn_model
        for train_traj_tr in self.dataset_batch :
            inputs_tr = train_traj_tr[0]
            break
        _ = self.model(inputs_tr)

        # self.dataset_batch_signature = self.get_dataset_signature(self.dataset_batch)

        ## load model
        if(args.load_model == True):
            load_model_path = os.path.join( CURRENT_DIR_PATH, args.load_model_path)
            check_model(self.model)            
            load_model(self.model, load_model_path )
            check_model(self.model)

    def _set_optimizer(self):
        self.optimizer = snt.optimizers.Adam(self.learning_rate) 

# ============ ============ ============ ============ ============

class TorqueErrorModel(TrainModel):
    def __init__(self, args):
        super().__init__(args, "TorqueErrorModel")

    def run(self):             
        # wrapper functions to be used
        def update_step(inputs_tr, targets_tr):
            with tf.GradientTape() as tape:
                output_op_tr = self.model(inputs_tr)
                # Loss.
                loss_tr = self.create_loss_ops(targets_tr, output_op_tr)

            gradients = tape.gradient(loss_tr, self.model.trainable_variables)
            self.optimizer.apply(gradients, self.model.trainable_variables)
            return output_op_tr, loss_tr 

        def val_loss(inputs_tr, targets_tr):
            output_op_tr = self.model(inputs_tr)
            # Loss.
            loss_ops_tr = self.create_loss_ops(targets_tr, output_op_tr)
            loss_tr = tf.math.reduce_sum(loss_ops_tr) 
            return output_op_tr, loss_tr

        ## step functions
        compiled_update_step = tf.function(update_step, 
                                input_signature=self.dataset_batch_signature)
        compiled_val_loss = tf.function(val_loss, 
                                input_signature=self.dataset_batch_signature) 

        ## save input data for future
        for train_traj_tr in self.dataset_batch :
            inputs_batch_tr = train_traj_tr[0]
            # targets_batch_tr = train_traj_tr[1]
            # output_batch_tr = self.model(inputs_batch_tr)
            # self.print_loss_ops(targets_batch_tr, output_batch_tr)
            break
        for single_traj_tr in self.dataset:
            input_tr = single_traj_tr[0]
            break

        self.save_sample_input(inputs_batch_tr, "inputs_batch_tr")
        self.save_sample_input(input_tr, "input_tr")                                                    

        ## training
        print("============ start !!! =============")
        self.TOTAL_TIMER.reset()
        batch_loss_sum = run_one_epoch(compiled_val_loss, self.dataset_batch)
        self.min_loss = batch_loss_sum #0.02  
        print("T {:.1f}, Ltr of initial raw model = {}".format( 
                                self.TOTAL_TIMER.elapsed(), batch_loss_sum ) )

        break_count=0
        for epoch in range(0, self.epoch_size):
            self.update_dataset_batch() #shuffle
            batch_loss_sum = run_one_epoch(compiled_update_step, self.dataset_batch)
            self.logf.record_time_loss(self.TOTAL_TIMER.elapsed(), batch_loss_sum)

            if(self.validation_test):
                val_batch_loss_sum = run_one_epoch(compiled_val_loss, self.valdataset_batch) 
                self.logf_val.record_time_loss(self.TOTAL_TIMER.elapsed(), val_batch_loss_sum)
                print("T {:.1f}, epoch_iter = {:02d}, Ltr {:.4f}, ValLtr {:.4f}".format(
                        self.TOTAL_TIMER.elapsed(), epoch, batch_loss_sum, val_batch_loss_sum))

                if(val_batch_loss_sum*0.7 > batch_loss_sum):
                    break_count = break_count + 1
                    print(" cnt = {:02d}, val_batch_loss_sum*0.7 > batch_loss_sum ", break_count)
                else:
                    break_count = 0
                
                if(break_count > 3):
                    print(" cnt = {:02d}", break_count)
                    break
                    
            else:
                print("T {:.1f}, epoch_iter = {:02d}, Ltr {:.4f}".format(
                    self.TOTAL_TIMER.elapsed(), epoch, batch_loss_sum))

            self.save_model(batch_loss_sum)   