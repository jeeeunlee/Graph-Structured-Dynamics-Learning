import tensorflow as tf
import numpy as np
import math
from graph_nets import utils_tf

from gn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import loadRobotURDF
from gn_inverse_dynamics.robot_graph_generator.robot_graph_generator import RobotGraphDataSetGenerator
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_definition import *
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator.hexamagneto_leg_definition import *
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator.hexamagneto_dynamic_graph_generator import HexaMagnetoLegDynamicGraphGenerator
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator.hexamagneto_leg_graph_base import HexaMagnetoLegGraphBase
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator.hexamagneto_static_graph import HexaMagnetoLegEdgeGraph

from gn_inverse_dynamics.utils.myutils import *
from gn_inverse_dynamics.utils.mymath import *
from gn_inverse_dynamics.robot_graph_generator.robot_graph import *

# log_with_time("RobotGraphDataSetGenerator")
# self.robot_graph_base = RobotGraph()
# self.data_size = None
# self.static_graph = None
# self.dynamic_graph_generator = DynamicGraphGenerator() 

class PassThresholdParam():
    def __init__(self, init_traj_pass_threshold=0, traj_pass_threshold=0):
        self.init_traj_pass_threshold = init_traj_pass_threshold
        self.traj_pass_threshold = traj_pass_threshold
        
#############################################################################
#   HexaMagnetoGraphGeneratorBase
#############################################################################
class HexaMagnetoGraphGeneratorBase(RobotGraphDataSetGenerator):
    def __init__(self, robot_graph_base, dynamic_graph_generator):
        self.data_size = None
        self.robot_graph_base = robot_graph_base # robot_base class        
        self.dynamic_graph_generator = dynamic_graph_generator # dyn generator class

    def get_path_str(self, path, print_prefix=''):
        log_with_time(" graph generator- {}: {}".format(print_prefix, path))
        if(type(path)!=str):
            path = path.decode('utf-8')
        return path 

#############################################################################
#   HexaMagnetoLegGraphGenerator
#############################################################################

class HexaMagnetoLegGraphGenerator(HexaMagnetoGraphGeneratorBase):
    def __init__(self, urdf_data_path, traj_data_path, pass_param=None):
        log_with_time("HexaMagnetoLegGraphGenerator")
        
        # check traj_data_path, urdf_data_path
        traj_data_path = self.get_path_str(traj_data_path,'traj_data_path')
        urdf_data_path = self.get_path_str(urdf_data_path,'urdf_data_path')
        leg_configs = self.get_leg_configs(urdf_data_path)
        
        # create dynamic graph generator
        dynamic_graph_generator = HexaMagnetoLegDynamicGraphGenerator(traj_data_path, leg_configs, pass_param)

        # create graph base
        robot_graph_base = HexaMagnetoLegGraphBase()

        # initialize objects
        super().__init__(robot_graph_base, dynamic_graph_generator)

    def get_leg_configs(self, urdf_data_path):
        robot = loadRobotURDF(urdf_data_path)
        leg_configs = dict()

        for leg in HexaMagnetoGraphEdge:
            for joint in robot.joints:
                if(joint.name == leg + '_coxa_joint'):
                    leg_configs[leg] = self.leg_config(joint)

        return leg_configs

    class leg_config():
        def __init__(self,joint):
            T_bi = joint.origin # array 4x4
            self.T_ib = inv_T(T_bi)
            self.p_ib = self.T_ib[0:3,3]
            self.R_ib = self.T_ib[0:3, 0:3]
            self.AdT_ib = AdT_from_T(self.T_ib)
            # print("leg_config({})".format(joint.name))
            # print(self.T_ib)
            # print(self.p_ib)
            # print(self.AdT_ib)

