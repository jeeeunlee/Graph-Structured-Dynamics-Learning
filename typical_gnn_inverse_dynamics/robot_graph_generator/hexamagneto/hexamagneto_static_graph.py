import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from typical_gnn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import  loadRobotURDF, LegData 
from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_graph_definition import HexaMagnetoGraphEdge, HexaMagnetoGraphNode
from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_joint_graph_base import HexaMagnetoJointGraphBase
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

from typical_gnn_inverse_dynamics.utils import mymath as mymath


# 4 leg type
def HexaMagnetoJointEdgeGraph(urdf_fn):
    ## Rotation Invariant model
    ## Setup : global, node, edge

    robot_graph_base = HexaMagnetoJointGraphBase()
    robot = loadRobotURDF(urdf_fn)
    
    edges = list()
    nodes = list()
     

    for joint_edge in HexaMagnetoGraphEdge:
        joint_data = list()
        for joint in robot.joints:
            if(joint.name == joint_edge):
                # print(joint.name)
                ang,pos = mymath.zyx_p_from_T(joint.origin)
                joint_data.extend(ang)
                joint_data.extend(pos)
                joint_data.extend(joint.axis)
                # print(joint_data)
        if(len(joint_data) < 9):
            print(joint_edge)
            edges.append([0]*9)
        else:
            edges.append(joint_data)

    for link_node in HexaMagnetoGraphNode:
        link_data = list()
        for link in robot.links:
            if(link.name == link_node):
                # print(link.name)
                link_data.append(link.inertial.mass)
                ang,pos = mymath.zyx_p_from_T(link.inertial.origin)
                link_data.extend(ang)
                link_data.extend(pos)                
                link_data.append(link.inertial.inertia[0][0])
                link_data.append(link.inertial.inertia[0][1])
                link_data.append(link.inertial.inertia[0][2])
                link_data.append(link.inertial.inertia[1][1])
                link_data.append(link.inertial.inertia[1][2])
                link_data.append(link.inertial.inertia[2][2])
                # print(link_data)
        if(len(link_data) < 13):
            print(link_node)
            nodes.append([0]*13)
        else:
            nodes.append(link_data)

    return robot_graph_base.generate_graph_dict([], nodes, edges)


        
# CHECK

# print("===========")
# urdf_path = os.path.join( CURRENT_DIR_PATH, 'typical_gnn_inverse_dynamics/robot_graph_generator/hexamagneto/magneto_hexa.urdf')

# robot = loadRobotURDF(urdf_path)

# for joint in robot.joints:
#     print(joint.name)
#     ang,pos = mymath.zyx_p_from_T(joint.origin)
#     print(ang)
#     print(pos)
#     print(joint.axis)

# for link in robot.links:
#     print(link.name)
#     ang,pos = mymath.zyx_p_from_T(link.inertial.origin)
#     print(ang)
#     print(pos)
#     print(link.inertial.mass)
#     print(link.inertial.inertia[0][0])
#     print(link.inertial.inertia[0][1])
#     print(link.inertial.inertia[0][2])
#     print(link.inertial.inertia[1][1])
#     print(link.inertial.inertia[1][2])
#     print(link.inertial.inertia[2][2])
    

# static_graph = HexaMagnetoJointEdgeGraph(urdf_path)

# gr_base.check_print(static_graph)
