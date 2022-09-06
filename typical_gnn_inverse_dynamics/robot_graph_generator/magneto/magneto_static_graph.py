import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from typical_gnn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import  loadRobotURDF 
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_graph_definition import *
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_joint_graph_base import MagnetoJointGraphBase
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

# 12 Joint
def MagnetoJointEdgeGraph(urdf_fn):
    ## Rotation Invariant model
    ## Setup : global, node, edge

    robot_graph_base = MagnetoJointGraphBase()
    robot = loadRobotURDF(urdf_fn)
    
    edges = list()
    nodes = list()
    global_features = list()

    for joint in robot.joints:

    for joint_edge in MagnetoGraphEdge:

    for link in robot.links:

    for link_node in MagnetoGraphNode:


    return robot_graph_base.generate_graph_dict(global_features, nodes, edges)

        
# CHECK

# print("===========")
# urdf_path = os.path.join( CURRENT_DIR_PATH, 'typical_gnn_inverse_dynamics/robot_graph_generator/magneto/magneto_simple.urdf')

# static_graph = MagnetoLegEdgeGraph(urdf_path)

# gr_base.check_print(static_graph)
