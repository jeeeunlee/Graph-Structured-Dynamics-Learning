import os, getpass
from collections import deque
import tensorflow as tf
import numpy as np
import math

from gn_inverse_dynamics.robot_graph_generator.hexamagneto import hexamagneto_definition as HexaMagneto
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator import hexamagneto_leg_definition as HexaMagnetoLegGraph

from gn_inverse_dynamics.utils import myutils as myutils
from gn_inverse_dynamics.utils import mymath as mymath

from gn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import *

class HexaMagnetoJointTrajDataGenerator():
    def __init__(self, traj_data_path, leg_configs, pass_param=None):
        myutils.log_with_time("HexaMagnetoJointTrajDataGenerator")
        self.set_traj_folders(traj_data_path)
        self.leg_configs = leg_configs
        self.data_size = 0
        if(pass_param):
            self.init_traj_pass_threshold = pass_param.init_traj_pass_threshold
            self.traj_pass_threshold = pass_param.traj_pass_threshold
        else:
            self.init_traj_pass_threshold = 0
            self.traj_pass_threshold = 0

    def get_data_size(self):
        return self.data_size 

    def gen_data_list(self):
        self.data_size = 0
        for traj_dict in self.gen_traj_dict():
            inputdata, targetdata = self.traj_dict_to_data_list(traj_dict)
            # input = concat(static, dynamic)
            self.data_size = self.data_size+1
            yield inputdata, targetdata

    def traj_dict_to_data_list(self, traj_dict):
        inputdatalist = list()
        targetdatalist = list()

        # grabity in base link frame
        # RZYX = traj_dict['q'][3:6] #RAD
        # R_wb = mymath.zyx_to_rot(RZYX)
        # R_bw = mymath.inv_R(R_wb)
        quat = traj_dict['rot_base'] #RAD
        R_wb = mymath.quat_to_rot(quat)
        R_bw = mymath.inv_R(R_wb)
        gw = [0.,0.,-9.8]
        gb = np.matmul(R_bw, gw).tolist()
        wb = traj_dict['base_body_vel'][0:3]

        for legname in HexaMagnetoLegGraph.HexaMagnetoGraphEdge:
            leg_data = list()
            leg_trq = list()

            # joint info
            leg_data.extend( np.matmul(self.leg_configs[legname].R_ib, gb).tolist() ) # 3
            leg_data.extend( np.matmul(self.leg_configs[legname].R_ib, wb).tolist() ) # 3   
            leg_data.extend( self.leg_configs[legname].p_ib ) # 3
            # leg_data.extend( np.matmul(self.leg_configs[legname].AdT_ib, base_vel).tolist() ) # 6

            magdataname = "mag_{}".format(legname.lower())
            cnctdataname = "ct_{}".format(legname.lower())
            leg_data.append( traj_dict[magdataname][0] )
            leg_data.append( traj_dict[cnctdataname][0] )
            
            for legjointname in ['coxa', 'femur', 'tibia']:
                jointname = "{}_{}_joint".format(legname, legjointname)
                leg_data.append(traj_dict['q'][ HexaMagneto.HexaMagnetoJoint[jointname] ]) # 3
                leg_data.append(traj_dict['dq'][ HexaMagneto.HexaMagnetoJoint[jointname] ]) # 3
                if('q_des' in traj_dict):
                    leg_data.append(traj_dict['q_des'][ HexaMagneto.HexaMagnetoJoint[jointname] ]) # 3
                    leg_data.append(traj_dict['dq_des'][ HexaMagneto.HexaMagnetoJoint[jointname] ]) # 3                     
                leg_trq.append( traj_dict['trq'][ HexaMagneto.HexaMagnetoJoint[jointname] ] )

            inputdatalist.append( leg_data )
            targetdatalist.append( leg_trq )
            
        return inputdatalist, targetdatalist


    def set_traj_folders(self, traj_data_dir):
        self.traj_data_dir = traj_data_dir
        self.traj_folders = os.listdir(traj_data_dir)
        self.folder_count = len(self.traj_folders)

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
        f_rotbase = open(file_path + "/rot_base.txt")

        f_mag_al = open(file_path + "/AL_mag_onoff.txt")
        f_mag_ar = open(file_path + "/AR_mag_onoff.txt")
        f_mag_bl = open(file_path + "/BL_mag_onoff.txt")
        f_mag_br = open(file_path + "/BR_mag_onoff.txt")
        f_mag_cl = open(file_path + "/CL_mag_onoff.txt")
        f_mag_cr = open(file_path + "/CR_mag_onoff.txt")

        f_ct_al = open(file_path + "/AL_contact_onoff.txt")
        f_ct_ar = open(file_path + "/AR_contact_onoff.txt")
        f_ct_bl = open(file_path + "/BL_contact_onoff.txt")
        f_ct_br = open(file_path + "/BR_contact_onoff.txt")
        f_ct_cl = open(file_path + "/CL_contact_onoff.txt")
        f_ct_cr = open(file_path + "/CR_contact_onoff.txt")

        f_base_body_vel = open(file_path + "/base_body_vel.txt")

        traj_file_zip_key = ['q', 'q_des', 'dq', 'dq_des', 'trq',
                        'mag_al', 'mag_ar', 'mag_bl', 'mag_br', 'mag_cl', 'mag_cr',
                        'ct_al', 'ct_ar', 'ct_bl', 'ct_br', 'ct_cl', 'ct_cr',
                        'base_body_vel', 'rot_base']
        traj_file_list = [f_q, f_q_d, f_dq, f_dq_d, f_trq,
                        f_mag_al, f_mag_ar, f_mag_bl, f_mag_br, f_mag_cl, f_mag_cr,
                        f_ct_al, f_ct_ar, f_ct_bl, f_ct_br,f_ct_cl,f_ct_cr,
                        f_base_body_vel, f_rotbase]
        traj_file_zip = zip(f_q, f_q_d, f_dq, f_dq_d, f_trq,  
                        f_mag_al, f_mag_ar, f_mag_bl, f_mag_br, f_mag_cl, f_mag_cr,
                        f_ct_al, f_ct_ar, f_ct_bl, f_ct_br, f_ct_cl,f_ct_cr,
                        f_base_body_vel, f_rotbase)

        return traj_file_zip, traj_file_zip_key, traj_file_list
    