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

from typical_gnn_inverse_dynamics.check_model.robot_model_check import RobotModelCheckSimulator


from typical_gnn_inverse_dynamics.utils import mygraphutils as mygraphutils
from typical_gnn_inverse_dynamics.utils import myutils as myutils


from random import uniform as uniform
from graph_nets import utils_tf

import typical_gnn_inverse_dynamics.my_graph_nets.gn_models as models
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph_generator import *
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_graph_generator import *
from typical_gnn_inverse_dynamics.train_model.run_functions import *

'''
traj data generator --------> [ simulator ] --------> (next state)
        |        (current state)  ^                         |
        | (input)                 | (output)                |
        v                         |                         |
      model  <-----------------------------------------------       
'''

class InterfaceParams():
    def __init__(self):
        self.save_dir = "a_result/checkmodel"
        self.base_data_path = "typical_gnn_inverse_dynamics/robot_graph_generator/magneto/magneto_simple.urdf"
        self.traj_data_dir = "a_dataset/magneto/rawData_icra/case6"
        self.saved_model_path = "a_result/2021icra/case5_96_np3/saved_model"

        self.num_processing_steps = 3
        self.edge_output_size = 3 
        self.edge_latent_size = 96

class MagnetoModelCheckInterface(RobotModelCheckSimulator):
    def __init__(self, args, name="RobotModelCheckInterface"): 
        self.name = name
        self._set_logger(args)
        self._set_params(args) 

        self._set_model(args) # self.model
        self._set_data_generator(args) # self.data_generator

        self.log_info()

    def initialize(self):
        # self.gen_traj : traj_dict, input_graph_tsr, target_graph_tsr
        # self.gen_traj = self.data_generator.gen_traj_dicts_and_graph_tr()
        # traj_dict, _, _ = next(self.gen_traj, (None, None, None) )

        self.gen_traj = self.data_generator.gen_traj_dict()
        traj_dict = next(self.gen_traj, None)

        q_init = traj_dict["q"]
        dq_init = traj_dict["dq"]
        # q_init = traj_dict["q_des"]
        # dq_init = traj_dict["dq_des"]
        
        return q_init, dq_init

    # def getCommand(self, q, dq, contacts):
    #     traj_dict, input_graph_tsr, target_graph_tsr = next(self.gen_traj, (None, None, None))
    #     if(traj_dict is None):
    #         print("traj_dict is none")
    #         exit()
            
    #     traj_dict["q"] = q
    #     traj_dict["dq"] = dq
    #     self.update_state(traj_dict)
    #     adhesions = self.update_env(traj_dict)
    #     # torque = self.update_action(input_graph_tsr, target_graph_tsr)
    #     # torque = self.target_action(input_graph_tsr, target_graph_tsr)
    #     torque = self.pdcontrol_action(traj_dict)

    #     # print(adhesions)

    #     return torque, adhesions

    def getCommand(self, q, dq, contacts):
        traj_dict = next(self.gen_traj, None)
        if(traj_dict is None):
            print("traj_dict is none")
            exit()
                                    
        traj_dict["q"] = q
        traj_dict["dq"] = dq
        input_graph_tsr, target_graph_tsr = self.data_generator.traj_dict_to_graph_tuple(traj_dict)
        self.update_state(traj_dict)
        adhesions = self.update_env(traj_dict)
        torque = self.update_action(input_graph_tsr, target_graph_tsr)
        # torque = self.target_action(input_graph_tsr, target_graph_tsr)
        # torque = self.pdcontrol_action(traj_dict)

        # print(adhesions)

        return torque, adhesions
    


    ###############################################
    # robot data analysis 
    def update_state(self, traj_dict):
        # extract q, qdot from traj_dict  
        q = traj_dict["q"]
        qdot = traj_dict["dq"]
        self.save_state(q, qdot)
        q_des = traj_dict["q_des"]
        qdot_des = traj_dict["dq_des"]
        self.save_cmd(q_des, qdot_des)
        pass

    def update_env(self, traj_dict):
        # extract magnetic force        
        adhesions = list()
        if(traj_dict['f_mag_al'][0] > 0.5):
            adhesions.append("AL_foot_link")
        if(traj_dict['f_mag_ar'][0] > 0.5):
            adhesions.append("AR_foot_link")
        if(traj_dict['f_mag_bl'][0] > 0.5):
            adhesions.append("BL_foot_link")
        if(traj_dict['f_mag_br'][0] > 0.5):
            adhesions.append("BR_foot_link")
        return adhesions

    def pdcontrol_action(self, traj_dict):
        q_d = traj_dict['q_des']
        dq_d = traj_dict['dq_des']

        q_s = traj_dict["q"]
        dq_s = traj_dict["dq"]

        trq = list()
        for qd, dqd, qs, dqs in zip(q_d, dq_d, q_s, dq_s):
            if( abs(qd) < 1e-3 and abs(dqd) < 1e-3):
                trq.append(0.0)
            else:
                val = 35.0 * (qd-qs) + 3.0 * (dqd-dqs)
                trq.append(val)

        return trq


    def target_action(self, input_graph_tsr, target_graph_tsr):
         
        target_edges = mygraphutils.edge_tensor_to_edge_numpy_list(target_graph_tsr.edges)

        target_action = []
        for trq in target_edges:
            target_action.extend(trq)
            target_action.extend([0.0, 0.0, 0.0])

        action = [0.0]*6 + target_action        
        return action


    def update_action(self, input_graph_tsr, target_graph_tsr):
        # myutils.log_with_time("update_action")
        output_graph_tsr = self.model(input_graph_tsr, self.num_processing_steps)
        # 3 dim edge to actions
        # myutils.log_with_time("model")

        # [[AL1,AL2,AL3], [BL1,BL2,..], ... ]
       
        output_edges = mygraphutils.edge_tensor_to_edge_numpy_list(output_graph_tsr[-1].edges) 
        target_edges = mygraphutils.edge_tensor_to_edge_numpy_list(target_graph_tsr.edges)

        output_action = []
        for trq in output_edges:
            output_action.extend(trq)
            output_action.extend([0.0, 0.0, 0.0])
        target_action = []
        for trq in target_edges:
            target_action.extend(trq)
            target_action.extend([0.0, 0.0, 0.0])

        self.save_trq_tg(target_action)
        self.save_trq_out(output_action)
        action = [0.0]*6 + output_action
        
        return action


    def set_env(self, traj_dict):
        mu = 0.7 # 0.7 + uniform(-0.1, 0.1)
        f_mag = 100 # 100 + uniform(-30.0, 30.0)
        self.agent.set_friction_coeff(mu)
        self.agent.set_magnetic_force(f_mag)

    def set_state(self, traj_dict):
        q = traj_dict["q"]
        qdot = traj_dict["dq"]
        self.agent.set_robot_state(q, qdot)        
        self.save_state(q, qdot)
        self.save_cmd(q, qdot)
        



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
        base_data_path = os.path.join(CURRENT_DIR_PATH, args.base_data_path)
        traj_data_dir = os.path.join(CURRENT_DIR_PATH, args.traj_data_dir)

        pass_param = PassThresholdParam(1000,0)

        self.data_generator = MagnetoLegGraphGenerator(
                        base_data_path = base_data_path,
                        traj_data_path = traj_data_dir,
                        pass_param = pass_param)        

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

