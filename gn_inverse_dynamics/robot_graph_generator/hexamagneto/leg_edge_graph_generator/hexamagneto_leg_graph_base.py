import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.robot_graph_generator.hexamagneto.leg_edge_graph_generator.hexamagneto_leg_definition import HexaMagnetoGraphEdgeSender,HexaMagnetoGraphEdgeReceiver
from gn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

######################################
#   DEFINE MAGNETO TOPOLOGY GRAPH    #
######################################
class HexaMagnetoLegGraphBase(RobotGraph):
    def __init__(self):
        receivers = []
        senders = []
        for sender in HexaMagnetoGraphEdgeSender:
            senders.append(HexaMagnetoGraphEdgeSender[sender]) 
        for receiver in HexaMagnetoGraphEdgeReceiver:
            receivers.append(HexaMagnetoGraphEdgeReceiver[receiver])         
        super().__init__(senders, receivers)
        
    # def generate_graph_dict(self, global_features, nodes, edges)
    # def concat_graph(self, graph_dicts)




