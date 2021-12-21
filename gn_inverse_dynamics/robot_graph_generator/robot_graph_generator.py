import os

import tensorflow as tf
import numpy as np
import math
import pickle

from gn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

from gn_inverse_dynamics.utils import mygraphutils as mygraphutils
from gn_inverse_dynamics.utils.myutils import * 
from gn_inverse_dynamics.utils import mymath as mymath

from graph_nets import utils_tf


############################################################
# robot graph G = (u,V,E) 
# G = concat(Gs, Gd)
# 
# Gs (static graph) : geometric properties of a robot
#                    ex) robot configuration / inertia properties
# Gd (dynamic grpah) : robot states
#                    ex) joint states / contact states / ext forces
############################################################

############################################################
#   dynamic graph generator - from trajectory
############################################################
class DynamicGraphGenerator():
    def __init__(self):
        pass
    def gen_dynamic_graph_dict(self):
        # yield dynamic_graph, target_graph
        pass
    def gen_traj_dict(self):
        # yield traj_dict
        pass    
    def traj_dict_to_graph_dicts(self, traj_dict):
        # return dynamic_graph, target_graph
        pass   
    def get_data_size(self):
        # return how many times generator is called
        pass

############################################################
#   graph dataset generator - from static / dynamic graph
############################################################
class RobotGraphDataSetGenerator():
    ## ======= ABSTRACT METHODS =======
    def __init__(self):
        log_with_time("RobotGraphDataSetGenerator")
        self.robot_graph_base = RobotGraph()
        self.data_size = None
        self.static_graph = None
        self.dynamic_graph_generator = DynamicGraphGenerator() 

    def gen_graph_dicts(self):
        # yield input_graph, target_graph
        for dynamic_graph, target_graph in self.dynamic_graph_generator.gen_dynamic_graph_dict():
            input_graph = self.robot_graph_base.concat_graph([dynamic_graph, self.static_graph])            
            yield input_graph, target_graph

    def gen_graph_tuples(self):
        for input_graph, target_graph in self.gen_graph_dicts():
            yield self.graph_dict_to_tuples(input_graph, target_graph)

    def graph_dict_to_tuples(self, input_graph, target_graph):
        target_graph_tsr = utils_tf.data_dicts_to_graphs_tuple( [target_graph] )
        input_graph_tsr = utils_tf.data_dicts_to_graphs_tuple( [input_graph] )
        # release casting problem by force casting 
        input_graph_tsr = input_graph_tsr.replace(
                        globals = tf.cast(input_graph_tsr.globals, tf.float32),
                        nodes = tf.cast(input_graph_tsr.nodes, tf.float32),
                        edges = tf.cast(input_graph_tsr.edges, tf.float32))
        return input_graph_tsr, target_graph_tsr

    def get_data_size(self):
        return self.dynamic_graph_generator.get_data_size()

    ## for check model interface
    def gen_traj_dicts_and_graph_tr(self):
        for traj_dict in self.dynamic_graph_generator.gen_traj_dict():
            input_graph_tsr, target_graph_tsr = self.traj_dict_to_graph_tuple(traj_dict)
            yield traj_dict, input_graph_tsr, target_graph_tsr

    def gen_traj_dict(self):
        for traj_dict in self.dynamic_graph_generator.gen_traj_dict():
            yield traj_dict

    def traj_dict_to_graph_tuple(self, traj_dict):
        dynamic_graph, target_graph = self.dynamic_graph_generator.traj_dict_to_graph_dicts(traj_dict)
        input_graph = self.robot_graph_base.concat_graph([dynamic_graph, self.static_graph]) 
        input_graph_tsr, target_graph_tsr = self.graph_dict_to_tuples(input_graph, target_graph)
        return input_graph_tsr, target_graph_tsr
            

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

    def get_full_tfdataset(self):
        gen = self.gen_graph_tuples()   
        graph_tuples_signature = mygraphutils.get_specs_from_graph_tuples(next(gen))
        tfdataset = tf.data.Dataset.from_generator(self.gen_graph_tuples,
                        output_signature = graph_tuples_signature)
        return tfdataset, tfdataset.element_spec

    def get_element_spec(self):
        tfdataset, element_spec = self.get_full_tfdataset()
        return element_spec