import sys
import collections
import itertools
import time
import os 

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

import pickle
import numpy as np
import tensorflow as tf
import sonnet as snt 

from gn_inverse_dynamics.utils import mygraphutils as mygraphutils
from gn_inverse_dynamics.utils import myutils as myutils


from random import uniform as uniform
from graph_nets import utils_tf

import gn_inverse_dynamics.my_graph_nets.gn_models as models
from gn_inverse_dynamics.robot_graph_generator.robot_graph_generator import *
from gn_inverse_dynamics.train_model.run_functions import *


'''
traj data generator --------> simulator--------> (next state)
        |        (current state)  ^ 
        | (input)                 | (output)
        v                         |
      model   ---------------------       
'''

class RobotModelCheckSimulator():
    def __init__(self, args, name="RobotModelCheckSimulator"): 
        self.name = name
        self._set_logger(args)
        self._set_params(args) 

        self._set_simulation() # self.agent
        self._set_model(args) # self.model
        self._set_data_generator(args) # self.data_generator

        self.log_info()

    def run(self):
        # for traj_dict, input_graph_tsr, target_graph_tsr in self.data_generator.gen_traj_dicts_and_graph_tr():
        #     self.update_state(traj_dict)
        #     self.update_env(traj_dict)
        #     self.update_action(input_graph_tsr, target_graph_tsr)
        #     q, qdot = self.agent.get_robot_state()
        pass


    ###############################################
    # robot data analysis 
    def update_state(self, traj_dict):
        # extract q, qdot from traj_dict        
        # self.agent.set_robot_state(q, qdot)        
        # self.save_state(q, qdot)
        # self.save_cmd(q, qdot)
        pass

    def update_action(self, input_graph_tsr, target_graph_tsr):
        # output_graph_tsr = self.model(input_graph_tsr, self.num_processing_steps)
        # output_action = mygraphutils.edge_tensor_to_edges_feature_list(output_graph_tsr[-1].edges)[0] # [[trq1, ..., ]]
        # target_action = mygraphutils.edge_tensor_to_edges_feature_list(target_graph_tsr[-1].edges)[0] # [[trq1, ..., ]]
        # self.save_trq_tg(target_action)
        # self.save_trq_out(output_action)
        # self.action = [0.0]*6 + output_action
        # self.agent.step(self.action)
        pass

    def update_env(self, traj_dict):
        # simulation environment settings
        # extract env params from traj_dict 
        # self.agent.set_magnet_onoff(contact_link_list)
        # self.agent.set_magnetic_force(force)
        # self.agent.set_friction_coeff(mu)
        pass



    ###############################################
    ## save data log
    def save_state(self, q, qdot):
        self.logf_q.record_list(q)
        self.logf_qdot.record_list(qdot)

    def save_cmd(self, q, qdot):
        self.logf_q_des.record_list(q)
        self.logf_dq_des.record_list(qdot)

    def save_trq_tg(self, trq):
        self.logf_trq_tg.record_list(trq)

    def save_trq_out(self, trq):
        self.logf_trq_out.record_list(trq)     

    ###############################################
    ## initial settings
    def _set_data_generator(self, args):
        self.data_generator = None        

    def _set_simulation(self):
        self.agent = None        

    def _set_logger(self, args):
        self.save_dir = os.path.join(CURRENT_DIR_PATH, args.save_dir)
        self.log_dir_path = os.path.join(self.save_dir, get_local_time() )
        
        myutils.create_folder(self.log_dir_path)
        self.logf_info = myutils.Logger(self.log_dir_path + '/info.csv')
        self.logf_q = myutils.Logger(self.log_dir_path + '/state_q.csv')
        self.logf_qdot = myutils.Logger(self.log_dir_path + '/state_qdot.csv')
        self.logf_q_des = myutils.Logger(self.log_dir_path + '/cmd_q.csv')
        self.logf_dq_des = myutils.Logger(self.log_dir_path + '/cmd_qdot.csv')
        self.logf_trq_tg = myutils.Logger(self.log_dir_path + '/trq_tg.csv')
        self.logf_trq_out = myutils.Logger(self.log_dir_path + '/trq_out.csv')

    def log_info(self):
        self.logf_info.record_string("saved_model_path={}".format(self.saved_model_path))
        self.logf_info.record_string("base_data_path={}".format(self.base_data_path))
        self.logf_info.record_string("traj_data_dir={}".format(self.traj_data_dir))        

    def _set_params(self, args):
        self.num_processing_steps = args.num_processing_steps
        self.init_traj_pass_threshold = 1000

        self.edge_latent_size = args.edge_latent_size
        self.edge_output_size = args.edge_output_size

        self.base_data_path = args.base_data_path
        self.traj_data_dir = args.traj_data_dir
    
    def _set_model(self, args):        
        self.saved_model_path = os.path.join(CURRENT_DIR_PATH, args.saved_model_path)        

        ## define gn_model
        self.model = models.EncodeProcessDecode(edge_latent_size = args.edge_latent_size, 
                                                edge_output_size = args.edge_output_size)
        ## initialize gn_model
        with open(self.saved_model_path + '/input_tr', 'rb') as in_:
            sample_input_tr = pickle.load(in_)
        print("==========================")
        print(sample_input_tr)
        print("==========================")
        _ = self.model(sample_input_tr, self.num_processing_steps)

        ## load model
        loaded = tf.saved_model.load( self.saved_model_path )
        for tfvar_load in loaded.all_variables:
            # print("loaded model variable name : " + tfvar_load.name)
            for tfvar_model in self.model.variables:
                if tfvar_model.name == tfvar_load.name:
                    tfvar_model.assign(tfvar_load.value())

