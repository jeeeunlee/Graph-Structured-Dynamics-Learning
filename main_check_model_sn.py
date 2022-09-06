import collections
import itertools
import time
import os 
import sys
import pickle

import tensorflow as tf

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.utils import myutils as myutils

from sn_inverse_dynamics.run_functions import *
from sn_inverse_dynamics import sn_model as myModels


class CheckModel():
    def __init__(self, args, name="CheckModel"):
        self.name = name
        self._set_logger(args)
        self._set_params(args)
        self._set_dataset(args) # dataset, testdataset
        self._set_load_model(args) # model /save,load path
        self._log_train_info(args)


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

        
    def run(self):
        def val_loss(inputs_tr, targets_tr):
            output_ops_tr = self.model(inputs_tr)
            loss_ops_tr = self.create_loss_ops(targets_tr, output_ops_tr)
            loss_tr = tf.math.reduce_sum(loss_ops_tr)
            return output_ops_tr, loss_tr

        compiled_val_loss = tf.function(val_loss, 
                            input_signature=self.dataset_batch_signature) 

        for train_traj_tr in self.dataset_batch:
            output_op_tr, loss_tr = compiled_val_loss(train_traj_tr[0], train_traj_tr[1])       
            self.logf_train.record_val(loss_tr.numpy().tolist())
            # print(loss_tr.numpy().tolist())

        for train_traj_tr in self.testdataset_batch:
            output_op_tr, loss_tr = compiled_val_loss(train_traj_tr[0], train_traj_tr[1])       
            self.logf_test.record_val(loss_tr.numpy().tolist())

    def _set_params(self,args):        
        self.latent_size = args.latent_size
        self.output_size = args.output_size
        self.num_layer = args.num_layer
        self.batch_size = args.batch_size
        self.num_processing_steps = args.num_processing_steps

    def _set_load_model(self, args):       
        ## define gn_model
        self.model = myModels.EncodeProcessDecode(latent_size=self.latent_size, 
                                                  output_size=self.output_size,
                                                  num_layer=self.num_layer,
                                                  num_processing_steps=self.num_processing_steps)
        ## initialize gn_model
        for train_traj_tr in self.dataset_batch :
            inputs_tr = train_traj_tr[0]
            #targets_tr = graph_reshape(train_traj_tr[1])
            break
        _ = self.model(inputs_tr)

         ## load model
        load_model_path = os.path.join( CURRENT_DIR_PATH, args.load_model_path)
        print(load_model_path)
        # check_model(self.model)            
        load_model(self.model, load_model_path )
        # check_model(self.model)

    def _set_logger(self, args):
        print(args.log_dir )
        if(args.log_dir == ''):
            log_dir = os.path.join("a_result", get_local_time() )
        else:
            log_dir = os.path.join("a_result", args.log_dir)
        log_dir_path = os.path.join(CURRENT_DIR_PATH, log_dir)
        create_folder(log_dir_path)
        self.log_dir_path = log_dir_path

        self.logf_info = Logger(log_dir_path + '/info.csv')
        self.logf_train = Logger(log_dir_path + '/train_error.csv')
        # self.logf_val = Logger(log_dir_path + '/validation_error.csv')
        self.logf_test = Logger(log_dir_path + '/validation_error.csv')

    def _log_train_info(self, args):
        self.logf_info.record_string("name = {}".format(self.name))
        self.logf_info.record_string("dataset = {}".format(self.dataset_path))
        self.logf_info.record_string("load_model_path = {}".format(args.load_model_path))
        self.logf_info.record_string("output_size no = {}".format(self.output_size))
        self.logf_info.record_string("layer no = {}".format(self.num_layer))
        self.logf_info.record_string("latent no = {}".format(self.latent_size))
        self.logf_info.record_string("num_processing_steps = {}".format(self.num_processing_steps))
        self.logf_info.record_string("batch size = {}".format(self.batch_size))        

    # SET DATA PATH
    def _set_dataset(self, args):
        self.dataset_path = os.path.join( CURRENT_DIR_PATH, args.dataset_path )
        self.test_dataset_path = os.path.join( CURRENT_DIR_PATH, args.test_dataset_path )


        with open(self.test_dataset_path + '/element_spec', 'rb') as in_:
            test_dataset_element_spec = pickle.load(in_)

        with open(self.dataset_path + '/element_spec', 'rb') as in_:
            dataset_element_spec = pickle.load(in_)
        

        # # load train dataset, validation dataset
        self.dataset = tf.data.experimental.load(self.dataset_path+"/fulldataset", element_spec=dataset_element_spec)
        self.testdataset = tf.data.experimental.load(self.test_dataset_path+"/fulldataset", element_spec=test_dataset_element_spec)

        # generate batch data
        self.dataset_batch = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.testdataset_batch = self.testdataset.batch(self.batch_size, drop_remainder=True)
        self.dataset_batch_signature = self.get_dataset_signature(self.dataset_batch)
   
    def get_dataset_signature(self, dataset):
        for data_batch in dataset:
            return ( tf.TensorSpec.from_tensor(data_batch[0]),
                        tf.TensorSpec.from_tensor(data_batch[1]) )    


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    # log directory
    parser.add_argument("--log_dir", type=str, default='')
    # graph-nets load/save
    parser.add_argument("--load_model_path", type=str, default="")
    # dataset path
    parser.add_argument("--dataset_path", type=str, default="a_dataset/tfData/ros_pnc/SN/case1")
    parser.add_argument("--test_dataset_path", type=str, default="a_dataset/tfData/ros_pnc/SN/case2")
    
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--output_size", type=int, default=3)    
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--num_processing_steps", type=int, default=2)
    

    args = parser.parse_args()
    mdl_cker = CheckModel(args)
    mdl_cker.run()