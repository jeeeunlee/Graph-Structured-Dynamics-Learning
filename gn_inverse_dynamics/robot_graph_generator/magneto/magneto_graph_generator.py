import tensorflow as tf
import numpy as np
import math
from graph_nets import utils_tf

from gn_inverse_dynamics.robot_graph_generator.robot_graph_generator import RobotGraphDataSetGenerator
from gn_inverse_dynamics.robot_graph_generator.magneto.magneto_definition import *
from gn_inverse_dynamics.robot_graph_generator.magneto.leg_edge_graph_generator.magneto_leg_definition import *
from gn_inverse_dynamics.robot_graph_generator.magneto.leg_edge_graph_generator.magneto_dynamic_graph_generator import MagnetoLegDynamicGraphGenerator
from gn_inverse_dynamics.robot_graph_generator.magneto.leg_edge_graph_generator.magneto_leg_graph_base import MagnetoLegGraphBase
from gn_inverse_dynamics.robot_graph_generator.magneto.leg_edge_graph_generator.magneto_static_graph import MagnetoLegEdgeGraph

from gn_inverse_dynamics.utils.myutils import *
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
#   MagnetoGraphGeneratorBase
#############################################################################
class MagnetoGraphGeneratorBase(RobotGraphDataSetGenerator):
    def __init__(self, robot_graph_base, static_graph, dynamic_graph_generator):
        self.data_size = None
        self.robot_graph_base = robot_graph_base # robot_base class        
        self.static_graph = static_graph # graph dict
        self.dynamic_graph_generator = dynamic_graph_generator # dyn generator class

    def get_path_str(self, path, print_prefix=''):
        log_with_time(" graph generator- {}: {}".format(print_prefix, path))
        if(type(path)!=str):
            path = path.decode('utf-8')
        return path 

#############################################################################
#   MagnetoLegGraphGenerator
#############################################################################

class MagnetoLegGraphGenerator(MagnetoGraphGeneratorBase):
    def __init__(self, base_data_path, traj_data_path, pass_param=None):
        log_with_time("MagnetoLegGraphGenerator")
        
        # check traj_data_path, base_data_path
        traj_data_path = self.get_path_str(traj_data_path,'traj_data_path')
        base_data_path = self.get_path_str(base_data_path,'base_data_path')

        # create static graph dict
        static_graph = MagnetoLegEdgeGraph(base_data_path)
        
        # create dynamic graph generator
        dynamic_graph_generator = MagnetoLegDynamicGraphGenerator(traj_data_path, pass_param)

        # create graph base
        robot_graph_base = MagnetoLegGraphBase()        

        # initialize objects
        super().__init__(robot_graph_base, static_graph, dynamic_graph_generator)

    # def gen_graph_dicts(self):
    #     for input_graph, target_graph in super().gen_graph_dicts():
    #         print("input_graph")
    #         self.robot_graph_base.check_print(input_graph)
    #         print("target_graph")
    #         self.robot_graph_base.check_print(target_graph)
    #         yield input_graph, target_graph


#############################################################################
#   OnlyDynamicDataGenerator
#############################################################################

# class OnlyDynamicDataGenerator(MagnetoGraphGeneratorBase):
#     def __init__(self, traj_data_path):
#         # no static graph
#         log_with_time("OnlyDynamicDataGenerator")
#         traj_data_path = self.get_path_str(traj_data_path,'traj_data_path')

#         robot_graph_base = MagnetoGraphBase()        
#         dynamic_graph_generator = MagnetoSingleTimeTrajectoryGraphGenerator(
#                                     robot_graph_base, traj_data_path)        
#         super().__init__(robot_graph_base, None, dynamic_graph_generator)
        
#     # abstract methods
#     def gen_graph_dicts(self):
#         # only dynamic graph without concatenation
#         for dynamic_graph, target_graph in self.dynamic_graph_generator.gen_dynamic_graph_dict():
#             yield dynamic_graph, target_graph

