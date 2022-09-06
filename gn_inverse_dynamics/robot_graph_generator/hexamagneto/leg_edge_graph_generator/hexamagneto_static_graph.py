import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.robot_graph_generator.load_data_from_urdf import  loadRobotURDF, LegData 
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator.hexamagneto_leg_definition import HexaMagnetoGraphEdge
from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator.hexamagneto_leg_graph_base import HexaMagnetoLegGraphBase
from gn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

# 4 leg type
def HexaMagnetoLegEdgeGraph(urdf_fn):
    ## Rotation Invariant model
    ## Setup : global, node, edge

    robot_graph_base = HexaMagnetoLegGraphBase()
    robot = loadRobotURDF(urdf_fn)
    

    # joint btw legs and baselink
    leg_joint_dict = dict()
    leg_neighborhood_dict = dict()
    for leg in HexaMagnetoGraphEdge:
        # print(leg)
        for joint in robot.joints:
            if(joint.name == leg + '_coxa_joint'):
                leg_joint_dict[leg] = joint

        if(leg=='AL'):
            leg_neighborhood_dict[leg] = ['AR', 'BL']
        elif(leg=='BL'):
            leg_neighborhood_dict[leg] = ['AL', 'BR']
        elif(leg=='BR'):
            leg_neighborhood_dict[leg] = ['BL', 'AR']
        elif(leg=='AR'):
            leg_neighborhood_dict[leg] = ['BR', 'AL']

    edges = list()
    for leg in HexaMagnetoGraphEdge:
        joint = leg_joint_dict[leg]
        leg1, leg2 = leg_neighborhood_dict[leg]
        joint1 = leg_joint_dict[leg1]
        joint2 = leg_joint_dict[leg2]

        edges.append( LegData(joint, joint1, joint2).extract_data() ) #get_data()


    return robot_graph_base.generate_graph_dict([], [], edges)

        
# CHECK

# print("===========")
# urdf_path = os.path.join( CURRENT_DIR_PATH, 'gn_inverse_dynamics/robot_graph_generator/hexamagneto/hexamagneto_simple.urdf')

# static_graph = HexaMagnetoLegEdgeGraph(urdf_path)

# gr_base.check_print(static_graph)
