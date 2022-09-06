import tensorflow as tf
import numpy as np
import math
from graph_nets import utils_tf

from typical_gnn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import loadRobotURDF
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph_generator import RobotGraphDataSetGenerator
from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_definition import *
from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_graph_definition import *
from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_dynamic_graph_generator import HexaMagnetoDynamicGraphGenerator
from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_joint_graph_base import HexaMagnetoJointGraphBase
from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_static_graph import HexaMagnetoJointEdgeGraph

from typical_gnn_inverse_dynamics.utils.myutils import *
from typical_gnn_inverse_dynamics.utils.mymath import *
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph import *

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
    def __init__(self, robot_graph_base, static_graph, dynamic_graph_generator):
        self.data_size = None
        self.robot_graph_base = robot_graph_base # robot_base class 
        self.static_graph = static_graph       
        self.dynamic_graph_generator = dynamic_graph_generator # dyn generator class        

    def get_path_str(self, path, print_prefix=''):
        log_with_time(" graph generator- {}: {}".format(print_prefix, path))
        if(type(path)!=str):
            path = path.decode('utf-8')
        return path 

#############################################################################
#   HexaMagnetoLegGraphGenerator
#############################################################################

class HexaMagnetoGraphGenerator(HexaMagnetoGraphGeneratorBase):
    def __init__(self, urdf_data_path, traj_data_path, pass_param=None):
        log_with_time("HexaMagnetoGraphGenerator")
        
        # check traj_data_path, urdf_data_path
        traj_data_path = self.get_path_str(traj_data_path,'traj_data_path')
        urdf_data_path = self.get_path_str(urdf_data_path,'urdf_data_path')

        # create static graph
        static_graph = HexaMagnetoJointEdgeGraph(urdf_data_path)
        
        # create dynamic graph generator
        dynamic_graph_generator = HexaMagnetoDynamicGraphGenerator(traj_data_path, pass_param)

        # create graph base
        robot_graph_base = HexaMagnetoJointGraphBase()

        # initialize objects
        super().__init__(robot_graph_base, static_graph, dynamic_graph_generator)