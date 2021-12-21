from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time
import os 

import numpy as np
import tensorflow as tf
import sonnet as snt 

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from gn_inverse_dynamics.my_graph_nets import gn_models as myModels

from gn_inverse_dynamics.utils.mygraphutils import *
from gn_inverse_dynamics.utils.myutils import *

###########################################################

def check_model(model):
    for tfvar_model in model.variables:
      if "EncodeProcessDecode/MLPGraphNetwork/graph_network/edge_block/mlp/linear_0/b" in tfvar_model.name:
        print(type(tfvar_model))
        print(tfvar_model)

def load_model(model, saved_model_path):
    loaded = tf.saved_model.load( saved_model_path )
    for tfvar_load in loaded.all_variables:
        # print("loaded model variable name : " + tfvar_load.name)
        for tfvar_model in model.variables:
            if tfvar_model.name == tfvar_load.name:
                tfvar_model.assign(tfvar_load.value())
                # print(tfvar_load.name + " is changed")
    return model

###########################################################

def get_dataset(dataset_path, element_spec, datasettype):
    if(datasettype == 'train'):
        datasetdir = 'traindataset'
    elif(datasettype == 'validation' or datasettype == 'val'):
        datasetdir = 'validationdataset'
    elif(datasettype == 'test'):
        datasetdir = 'testdataset'
    elif(datasettype == 'full'):
        datasetdir = 'fulldataset'
    else:
        datasetdir = datasettype
    dataset_path = os.path.join( dataset_path, datasetdir )
    dataset = tf.data.experimental.load(dataset_path, element_spec=element_spec)
    return dataset

###########################################################

def run_one_epoch_reward(compiled_function, dataset_batch):
    batch_iter = 0
    batch_loss_sum = 0.
    for train_traj_tr in dataset_batch :
        # log_with_time("run_one_epoch")   
        inputs_tr = graph_reshape(train_traj_tr[0])
        targets_tr = graph_reshape(train_traj_tr[1])
        reward = train_traj_tr[2]
        # log_with_time("graph_reshape")   
        output_ops_tr, loss_tr = compiled_function(inputs_tr, targets_tr, reward)
        # log_with_time("compiled_function") 
        batch_loss_sum = batch_loss_sum + loss_tr
        batch_iter = batch_iter+1 
    batch_loss_sum = batch_loss_sum/batch_iter
    return batch_loss_sum

def run_one_epoch(compiled_function, dataset_batch):
    batch_iter = 0
    batch_loss_sum = 0.
    for train_traj_tr in dataset_batch :
        # log_with_time("run_one_epoch")   
        inputs_tr = graph_reshape(train_traj_tr[0])
        targets_tr = graph_reshape(train_traj_tr[1])
        # log_with_time("graph_reshape")   
        output_ops_tr, loss_tr = compiled_function(inputs_tr, targets_tr)
        # log_with_time("compiled_function") 
        batch_loss_sum = batch_loss_sum + loss_tr
        batch_iter = batch_iter+1 
    batch_loss_sum = batch_loss_sum/batch_iter
    return batch_loss_sum

###########################################################
def save_graph_edge(input_graph_tuples, f):
  n_graph = utils_tf.get_num_graphs(input_graph_tuples)
  # print("there is {:02d} graph".format(n_graph))

  for i in range(n_graph):
    sliced_graph = utils_tf.get_graph(input_graph_tuples,i)
    edge_tensor = sliced_graph.edges
    edge_list = edge_tensor.numpy()

    for edge in edge_list:
      f.write(str(edge[0]))
      f.write(' ')

    f.write('\n')



# DIFF FUCTION
# @tf.function(input_signature=traj_signature)
# def diff_model(inputs_tr, targets_tr):
#   output_ops_tr = model(inputs_tr, num_processing_steps_tr)
#   diff_ops_tr = [ targets_tr.edges - output_op_tr.edges
#                   for output_op_tr in output_ops_tr ]
#   return diff_ops_tr

# for diff_op_tr in diff_ops_tr:
#   print("-----------------------")
#   print(type(diff_op_tr))
#   print()
#   print(diff_op_tr)