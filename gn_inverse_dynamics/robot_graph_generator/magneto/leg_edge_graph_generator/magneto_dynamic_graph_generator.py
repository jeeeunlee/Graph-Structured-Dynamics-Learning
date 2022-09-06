import os, getpass
from collections import deque
import tensorflow as tf
import numpy as np
import math

from graph_nets import utils_tf

from gn_inverse_dynamics.utils import myutils as myutils
from gn_inverse_dynamics.utils import mymath as mymath
from gn_inverse_dynamics.utils import mygraphutils as mygraphutils

from gn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import *
from gn_inverse_dynamics.robot_graph_generator.robot_graph_generator import DynamicGraphGenerator

from gn_inverse_dynamics.robot_graph_generator.magneto import magneto_definition as Magneto
from gn_inverse_dynamics.robot_graph_generator.magneto.leg_edge_graph_generator import magneto_leg_definition as MagnetoLegGraph
from gn_inverse_dynamics.robot_graph_generator.magneto.leg_edge_graph_generator.magneto_leg_graph_base import MagnetoLegGraphBase



############################################################
#   dynamic graph generator - traj to dynamic graph
############################################################

# BASE GRAPH GENERATOR - common functions
class MagnetoLegDynamicGraphGenerator(DynamicGraphGenerator):
    def __init__(self, traj_data_path, leg_configs, pass_param=None):
        myutils.log_with_time("MagnetoLegDynamicGraphGenerator")
        self.robot_graph_base = MagnetoLegGraphBase()
        self.set_traj_folders(traj_data_path)
        self.data_size = 0
        self.leg_configs = leg_configs

        if(pass_param):
            self.init_traj_pass_threshold = pass_param.init_traj_pass_threshold
            self.traj_pass_threshold = pass_param.traj_pass_threshold
        else:
            self.init_traj_pass_threshold = 0
            self.traj_pass_threshold = 0

    def get_data_size(self):
        return self.data_size

    def gen_dynamic_graph_dict(self):
        self.data_size = 0
        for traj_dict in self.gen_traj_dict():
            dynamic_graph, target_graph = self.traj_dict_to_graph_dicts(traj_dict)
            # input = concat(static, dynamic)
            self.data_size = self.data_size+1
            yield dynamic_graph, target_graph

    # for model check
    def gen_traj_dict(self, init_traj_pass_threshold=None, traj_pass_threshold=None):
        if(init_traj_pass_threshold is None):
            init_traj_pass_threshold=self.init_traj_pass_threshold
        if(traj_pass_threshold is None):
            traj_pass_threshold=self.traj_pass_threshold

        self.data_size = 0
        for raw_traj_folder in self.traj_folders:
            traj_file_zip, traj_file_zip_key, _ = self.get_trajectory_files( self.traj_data_dir + '/' + raw_traj_folder )            
            i=0
            while(True):                
                ## pass trajectory
                if(i==0):
                    for _ in range(init_traj_pass_threshold):
                        traj_flines = next(traj_file_zip, None)
                else:
                    for _ in range(traj_pass_threshold):
                        traj_flines = next(traj_file_zip, None)

                # read current line
                traj_flines = next(traj_file_zip, None)

                # termination
                if(traj_flines==None):
                    break

                # create traj_dict
                traj_dict = dict()
                for traj_key, traj_fline in zip(traj_file_zip_key, traj_flines):
                    traj_dict[traj_key] = myutils.string_to_list(traj_fline)

                i = i+1
                yield traj_dict

    def set_traj_folders(self, traj_data_dir):
        self.traj_data_dir = traj_data_dir
        self.traj_folders = os.listdir(traj_data_dir)
        self.folder_count = len(self.traj_folders)
        # for raw_traj_folder in self.traj_folders:
        #     traj_data_path = self.traj_data_dir + '/' + raw_traj_folder
        #     print(traj_data_path) 

    def get_trajectory_files(self, file_path):
        # q, q_des, dotq, dotq_des, trq, contact_al, f_mag_al, base_ori
        f_q = open(file_path + "/q_sen.txt")
        f_dq = open(file_path + "/qdot_sen.txt")
        # f_q_d = open(file_path + "/q_des.txt")
        # f_dq_d = open(file_path + "/qdot_des.txt")   
        f_q_d = open(file_path + "/q_sen.txt")
        f_dq_d = open(file_path + "/qdot_sen.txt")
        next(f_q_d)
        next(f_dq_d)      
        f_trq = open(file_path + "/trq.txt")

        f_mag_al = open(file_path + "/AL_mag_onoff.txt")
        f_mag_ar = open(file_path + "/AR_mag_onoff.txt")
        f_mag_bl = open(file_path + "/BL_mag_onoff.txt")
        f_mag_br = open(file_path + "/BR_mag_onoff.txt")

        f_ct_al = open(file_path + "/AL_contact_onoff.txt")
        f_ct_ar = open(file_path + "/AR_contact_onoff.txt")
        f_ct_bl = open(file_path + "/BL_contact_onoff.txt")
        f_ct_br = open(file_path + "/BR_contact_onoff.txt")

        f_base_body_vel = open(file_path + "/base_body_vel.txt")

        traj_file_zip_key = ['q', 'q_des', 'dq', 'dq_des', 'trq',
                        'mag_al', 'mag_ar', 'mag_bl', 'mag_br',
                        'ct_al', 'ct_ar', 'ct_bl', 'ct_br',
                        'base_body_vel']
        traj_file_list = [f_q, f_q_d, f_dq, f_dq_d, f_trq,
                        f_mag_al, f_mag_ar, f_mag_bl, f_mag_br,
                        f_ct_al, f_ct_ar, f_ct_bl, f_ct_br,
                        f_base_body_vel]
        traj_file_zip = zip(f_q, f_q_d, f_dq, f_dq_d, f_trq,  
                        f_mag_al, f_mag_ar, f_mag_bl, f_mag_br,
                        f_ct_al, f_ct_ar, f_ct_bl, f_ct_br ,
                        f_base_body_vel)

        return traj_file_zip, traj_file_zip_key, traj_file_list


    def traj_dict_to_graph_dicts(self, traj_dict):
        # dyn_nodes
        dyn_nodes = self.compute_nodes(traj_dict)       
        # dyn_edges, target_edges
        dyn_edges, target_edges = self.compute_edges(traj_dict)
        # dyn_globals
        dyn_globals = self.compute_globals(traj_dict)

        dynamic_graph = self.robot_graph_base.generate_graph_dict(dyn_globals, dyn_nodes, dyn_edges)
        target_graph = self.robot_graph_base.generate_graph_dict([], [], target_edges)

        return dynamic_graph, target_graph

    def compute_nodes(self, traj_dict):
        dyn_nodes=[]
        for linkname in MagnetoLegGraph.MagnetoGraphNode:
            if(linkname =='AR'):
                f_magnetic_onoff = traj_dict['mag_ar'][0]
                f_contact_onoff = traj_dict['ct_ar'][0]
                data = [ f_magnetic_onoff,  f_contact_onoff]
            elif(linkname =='BR'):
                f_magnetic_onoff = traj_dict['mag_br'][0]
                f_contact_onoff = traj_dict['ct_br'][0]
                data = [ f_magnetic_onoff,  f_contact_onoff]
            elif(linkname =='AL'):
                f_magnetic_onoff = traj_dict['mag_al'][0]
                f_contact_onoff = traj_dict['ct_al'][0]
                data = [ f_magnetic_onoff,  f_contact_onoff]
            elif(linkname =='BL'):
                f_magnetic_onoff = traj_dict['mag_bl'][0]
                f_contact_onoff = traj_dict['ct_bl'][0]
                data = [ f_magnetic_onoff,  f_contact_onoff]
            else:
                data = [ 0.0, 0.0 ]
            dyn_nodes.append(data)
        return dyn_nodes

    def compute_edges(self, traj_dict):
        dyn_edges=[]
        target_edges=[]

        # grabity in base link frame
        RZYX = traj_dict['q'][3:6] #RAD
        R_wb = mymath.zyx_to_rot(RZYX)
        R_bw = mymath.inv_R(R_wb)
        gw = [0.,0.,-9.8]
        gb = np.matmul(R_bw, gw).tolist()
        base_vel = traj_dict['base_body_vel']

        for legname in MagnetoLegGraph.MagnetoGraphEdge:
            leg_data = list()
            leg_trq = list()
            # joint info
            leg_data.extend( np.matmul(self.leg_configs[legname].R_ib, gb).tolist() ) # 3
            leg_data.extend( np.matmul(self.leg_configs[legname].AdT_ib, base_vel).tolist() ) # 6

            for legjointname in ['coxa', 'femur', 'tibia']:
                jointname = "{}_{}_joint".format(legname, legjointname)
                
                leg_data.append(traj_dict['q'][ Magneto.MagnetoJoint[jointname] ]) # 3
                leg_data.append(traj_dict['dq'][ Magneto.MagnetoJoint[jointname] ]) # 3
                if('q_des' in traj_dict):
                    leg_data.append(traj_dict['q_des'][ Magneto.MagnetoJoint[jointname] ]) # 3
                    leg_data.append(traj_dict['dq_des'][ Magneto.MagnetoJoint[jointname] ]) # 3          
                
                leg_trq.append( traj_dict['trq'][ Magneto.MagnetoJoint[jointname] ] ) #3

            dyn_edges.append( leg_data )
            target_edges.append( leg_trq )

        return dyn_edges, target_edges


    def compute_globals(self, traj_dict):
        dyn_globals=[0]

        return dyn_globals

