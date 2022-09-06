import tensorflow as tf
import numpy as np
import math
from graph_nets import utils_tf

from typical_gnn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import loadRobotURDF
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph_generator import RobotGraphDataSetGenerator
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_definition import *
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_graph_definition import *
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_dynamic_graph_generator import MagnetoDynamicGraphGenerator
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_static_graph import MagnetoJointEdgeGraph

from typical_gnn_inverse_dynamics.utils.myutils import *
from typical_gnn_inverse_dynamics.utils.mymath import *
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph import *

######################################
#   DEFINE MAGNETO TOPOLOGY GRAPH    #
######################################
class MagnetoGraphBase(RobotGraph):
    def __init__(self):
        receivers = []
        senders = []
        for sender in MagnetoGraphEdgeSender:
            senders.append(MagnetoGraphEdgeSender[sender]) 
        for receiver in MagnetoGraphEdgeReceiver:
            receivers.append(MagnetoGraphEdgeReceiver[receiver])         
        super().__init__(senders, receivers)
        
    # def generate_graph_dict(self, global_features, nodes, edges)
    # def concat_graph(self, graph_dicts)

class PassThresholdParam():
    def __init__(self, init_traj_pass_threshold=0, traj_pass_threshold=0):
        self.init_traj_pass_threshold = init_traj_pass_threshold
        self.traj_pass_threshold = traj_pass_threshold
        
#############################################################################
#   MagnetoGraphGeneratorBase
#############################################################################
class MagnetoGraphGeneratorBase(RobotGraphDataSetGenerator):
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
#   MagnetoLegGraphGenerator
#############################################################################

class MagnetoGraphGenerator(MagnetoGraphGeneratorBase):
    def __init__(self, urdf_data_path, traj_data_path, pass_param=None):
        log_with_time("MagnetoGraphGenerator")
        
        # check traj_data_path, urdf_data_path
        traj_data_path = self.get_path_str(traj_data_path,'traj_data_path')
        urdf_data_path = self.get_path_str(urdf_data_path,'urdf_data_path')
        
        # create dynamic graph generator
        dynamic_graph_generator = MagnetoDynamicGraphGenerator(traj_data_path, pass_param)

        # create graph base
        robot_graph_base = MagnetoGraphBase()

        # initialize objects
        super().__init__(robot_graph_base, dynamic_graph_generator)

