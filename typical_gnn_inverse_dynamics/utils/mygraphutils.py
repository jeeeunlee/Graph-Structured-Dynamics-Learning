from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import tensorflow as tf
from graph_nets import graphs
from graph_nets import utils_tf

import typical_gnn_inverse_dynamics.utils.mymath as mymath


#################### graph split and concat ####################

# NODES = "nodes"
# EDGES = "edges"
# RECEIVERS = "receivers"
# SENDERS = "senders"
# GLOBALS = "globals"
# N_NODE = "n_node"
# N_EDGE = "n_edge"
# ALL_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE)

def concat_graph_edge_to_batch_edge_tensor(graph_tuples_batch):
  batch_size = len(graph_tuples_batch.n_edge)
  return tf.split(value=graph_tuples_batch.edges, axis=0, num_or_size_splits=batch_size) 

def tensor_to_list(ts, axis=0):
  shape_list = ts.shape.as_list()
  split_size = shape_list[axis]
  ts_list = tf.split(value=ts, axis=axis, num_or_size_splits=split_size)
  return [ tf.squeeze(ts_split, [axis], name=None) for ts_split in ts_list]

def graph_split(graph_tuples_batch):
  # batched graph_tuples in shape dimension -> list
  nodes = tensor_to_list(graph_tuples_batch.nodes)
  edges = tensor_to_list(graph_tuples_batch.edges)
  globals_ = tensor_to_list(graph_tuples_batch.globals)
  receivers = tensor_to_list(graph_tuples_batch.receivers)
  senders = tensor_to_list(graph_tuples_batch.senders)
  n_node = tensor_to_list(graph_tuples_batch.n_node)
  n_edge = tensor_to_list(graph_tuples_batch.n_edge)
  n_graph = len(n_edge)
  return [
   graph_tuples_batch.replace(nodes=nodes[i], edges=edges[i], globals=globals_[i],
   receivers=receivers[i], senders=senders[i], n_node=n_node[i], n_edge=n_edge[i])
   for i in range(n_graph) ]

def concat_graph_split(graph_tuples_batch):
  # concatenated graph -> list
  batch_size = len(graph_tuples_batch.n_node)
  # graph_nodes : []
  nodes =  tf.split(value=graph_tuples_batch.nodes, axis=0, num_or_size_splits=batch_size)
  edges =  tf.split(value=graph_tuples_batch.edges, axis=0, num_or_size_splits=batch_size) 
  globals_ = tf.split(value=graph_tuples_batch.globals, axis=0, num_or_size_splits=batch_size) 
  receivers = tf.split(value=graph_tuples_batch.receivers, axis=0, num_or_size_splits=batch_size) 
  senders = tf.split(value=graph_tuples_batch.senders, axis=0, num_or_size_splits=batch_size) 
  n_node = tf.split(value=graph_tuples_batch.n_node, axis=0, num_or_size_splits=batch_size) 
  n_edge = tf.split(value=graph_tuples_batch.n_edge, axis=0, num_or_size_splits=batch_size) 

  return [
   graph_tuples_batch.replace(nodes=nodes[i], edges=edges[i], globals=globals_[i],
   receivers=receivers[i], senders=senders[i], n_node=n_node[i], n_edge=n_edge[i])
   for i in range(batch_size) ]

def graph_reshape(graph_tuple_batch):
  graph_lists = graph_split(graph_tuple_batch)
  concat_graph_tuples = utils_tf.concat(graph_lists, axis=0)
  return concat_graph_tuples

def reward_reshape(reward, batch_size):
  reward = tf.split(value = reward, axis=0, num_or_size_splits=batch_size)

def graph_tuples_reshape(graph_tuples_batch):
  reshaped_graph_list=[]
  for graph_tuple_batch in graph_tuples_batch:
    reshaped_graph_list.append(graph_reshape(graph_tuple_batch))
  return tuple(reshaped_graph_list)

def get_specs_from_graph_tuples_and_reward_batch_set(graph_tuples_with_reward_batch_set):
  for graph_tuples_with_reward_batch in graph_tuples_with_reward_batch_set:
    concat_input = graph_reshape(graph_tuples_with_reward_batch[0])
    concat_output = graph_reshape(graph_tuples_with_reward_batch[1])
    reward_tensor = graph_tuples_with_reward_batch[2]
    break

    
  graph_input_spec = utils_tf.specs_from_graphs_tuple(concat_input)
  graph_target_spec = utils_tf.specs_from_graphs_tuple(concat_output)
  reward_spec = tf.TensorSpec.from_tensor(reward_tensor)
  # print("get_specs_from_graph_tuples_and_reward")
  # print(graph_input_spec)
  # print(graph_target_spec)
  # print(reward_spec)
  return (graph_input_spec, graph_target_spec, reward_spec) 

def get_specs_from_graph_tuples_and_reward(graph_tuples_with_reward):
  graph_input_spec = utils_tf.specs_from_graphs_tuple(graph_tuples_with_reward[0])
  graph_target_spec = utils_tf.specs_from_graphs_tuple(graph_tuples_with_reward[1])
  reward_spec = tf.TensorSpec.from_tensor(graph_tuples_with_reward[2])
  # print("get_specs_from_graph_tuples_and_reward")
  # print(graph_input_spec)
  # print(graph_target_spec)
  # print(reward_spec)
  return (graph_input_spec, graph_target_spec, reward_spec)

def get_specs_from_graph_tuples(graph_tuples):
  # assumed that graph_tuples = (g_input, g_output)
  return (utils_tf.specs_from_graphs_tuple(graph_tuples[0]), 
            utils_tf.specs_from_graphs_tuple(graph_tuples[1]))

def get_specs_from_graph_tuples_batch_sets(graph_tuples_batch_sets):
  for graph_tuples_batch in graph_tuples_batch_sets:
    graph_tuples = graph_tuples_reshape(graph_tuples_batch)
    break
  return get_specs_from_graph_tuples(graph_tuples)

def get_batch_specs_from_graph_tuples_batch_sets(graph_tuples_batch_sets):
  for graph_tuples_batch in graph_tuples_batch_sets:
    break
  return get_specs_from_graph_tuples(graph_tuples_batch)

def get_type_shape_from_dict(example_dict):
  traj_type={}
  traj_shape={}
  for key in example_dict:
    traj = example_dict[key]
    traj_type[key] = traj.dtype
    traj_shape[key] = traj.shape
  # print(example_dict)
  # print(traj_type)
  # print(traj_shape)
  return traj_type, traj_shape

def get_type_shape_from_dicts(example_dicts):
  traj_types = []
  traj_shapes = []
  for example_dict in example_dicts:
    print(example_dict)
    traj_type, traj_shape = get_type_shape_from_dict(example_dict)
    traj_types.append(traj_type)
    traj_shapes.append(traj_shape)
  return traj_types, traj_shapes


###################################################################################
# edge_tensor : [[a1,b1,c1,,,, m1], [a2,b2,c2,,,], ]: feature dim = m
# edge_numpy_list : [[a1,b1,c1,,,, m1], [a2,b2,c2,,,], ]: feature dim = m
# edges_feature_list : [[a1,a2,,,,an], [b1,b2,,,], [c1,c2,,,], ]: edge dim = n

def edge_tensor_to_edge_numpy_list(edge_tensor):
  edge_numpy_list = (edge_tensor.numpy()).tolist()
  return edge_numpy_list

def edge_tensor_to_edges_feature_list(edge_tensor):
  edge_numpy = (edge_tensor.numpy()).tolist()
  edges_feature_list = list()
  for edge_feature in edge_numpy: # edge_feature = [ai,bi,ci,,,mi] of edge i
    for i, edge_element in enumerate(edge_feature):
      if(len(edges_feature_list)<len(edge_feature)):
        edges_feature_list.append([edge_element])
      else:
        edges_feature_list[i].append(edge_element)
  return edges_feature_list

def edges_feature_list_to_edge_tensor(edges_feature_list):
  edge_tensor = list()
  for edges_feature in edges_feature_list:
    for i, qi in enumerate(edges_feature):
      if(len(edges_feature_list)<len(edges_feature)):
        edge_tensor.append([qi])
      else:
        edge_tensor[i].append(qi)
  return tf.convert_to_tensor(edge_tensor, dtype=tf.float32)

def graph_tensor_to_numpy(graph_tensor):
  return graph_tensor.replace(
    nodes = graph_tensor.nodes.numpy(),
    edges = graph_tensor.edges.numpy(),
    globals = graph_tensor.globals.numpy() )

def graph_numpy_to_tensor(graph_numpy):
  return graph_numpy.replace(
    nodes = tf.convert_to_tensor(graph_numpy.nodes, dtype=tf.float32),
    edges = tf.convert_to_tensor(graph_numpy.edges, dtype=tf.float32),
    globals = tf.convert_to_tensor(graph_numpy.globals, dtype=tf.float32) )
