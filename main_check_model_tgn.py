import collections
import itertools
import time
import os 
import sys
import pickle

import tensorflow as tf

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from typical_gnn_inverse_dynamics.utils import myutils as myutils
from typical_gnn_inverse_dynamics.utils.mygraphutils import *
from typical_gnn_inverse_dynamics.train_model.run_functions import *
from typical_gnn_inverse_dynamics.my_graph_nets import gn_models as myModels


class CheckModel():
    def __init__(self, args, name="CheckModel"):
        self.name = name
        self._set_logger(args)
        self._set_params(args)
        self._set_dataset(args) # dataset, testdataset
        self._set_load_model(args) # model /save,load path
        self._log_train_info(args)  


    def create_loss_ops(self, target_op, output_ops):
        ''' Create supervised loss operations from targets and outputs.
        Args:
            target_op: The tensor of target torque (edge).
            output_ops: The list of output graphs from the model.

        Returns:
            A list of loss values (tf.Tensor), one per output op.
        '''
        loss_ops = [tf.reduce_mean( tf.reduce_mean((output_op.edges - target_op.edges)**2, axis=-1))
                    for output_op in output_ops   ]

       
        return loss_ops

        
    def run(self):
        def val_loss(inputs_tr, targets_tr):
            output_ops_tr = self.model(inputs_tr, self.num_processing_steps)
            loss_ops_tr = self.create_loss_ops(targets_tr, output_ops_tr)
            #loss_tr = tf.math.reduce_sum(loss_ops_tr) / self.num_processing_steps
            loss_tr = loss_ops_tr[self.num_processing_steps-1]
            return output_ops_tr, loss_tr

        compiled_val_loss = tf.function(val_loss, 
                            input_signature=self.dataset_batch_signature) 

        for train_traj_tr in self.dataset_batch:
            inputs_tr = graph_reshape(train_traj_tr[0])
            targets_tr = graph_reshape(train_traj_tr[1])
            output_ops_tr, loss_tr = compiled_val_loss(inputs_tr, targets_tr)       
            self.logf_train.record_val(loss_tr.numpy().tolist())
            # print(loss_tr.numpy().tolist())

        for train_traj_tr in self.testdataset_batch:
            inputs_tr = graph_reshape(train_traj_tr[0])
            targets_tr = graph_reshape(train_traj_tr[1])
            output_ops_tr, loss_tr = compiled_val_loss(inputs_tr, targets_tr)       
            self.logf_test.record_val(loss_tr.numpy().tolist())

    def _set_params(self,args):
        self.num_processing_steps = args.num_processing_steps
        self.edge_latent_size = args.edge_latent_size
        self.edge_output_size = args.edge_output_size
        self.batch_size = args.batch_size      

    def _set_load_model(self, args):       
        ## define gn_model
        self.model = myModels.EncodeProcessDecode(edge_latent_size=self.edge_latent_size, 
                                                  edge_output_size=self.edge_output_size)
        ## initialize gn_model
        for train_traj_tr in self.dataset_batch :
            inputs_tr = graph_reshape(train_traj_tr[0])
            #targets_tr = graph_reshape(train_traj_tr[1])
            break
        _ = self.model(inputs_tr, self.num_processing_steps)

         ## load model
        load_model_path = os.path.join( CURRENT_DIR_PATH, args.load_model_path)
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
        self.logf_info.record_string("edge_output_size no = {}".format(self.edge_output_size))
        self.logf_info.record_string("latent no = {}".format(self.edge_latent_size))
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
        self.dataset_batch_signature = get_specs_from_graph_tuples_batch_sets(self.dataset_batch)
   


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    # log directory
    parser.add_argument("--log_dir", type=str, default='')
    # graph-nets load/save
    parser.add_argument("--load_model_path", type=str, default="")
    # dataset path
    parser.add_argument("--dataset_path", type=str, default="a_dataset/magneto/tfData_slope_combination/case1")
    parser.add_argument("--test_dataset_path", type=str, default="a_dataset/magneto/tfData_slope_combination/case1_test")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--edge_latent_size", type=int, default=64)
    parser.add_argument("--num_processing_steps", type=int, default=3)
    parser.add_argument("--edge_output_size", type=int, default=1)    

    args = parser.parse_args()
    mdl_cker = CheckModel(args)
    mdl_cker.run()