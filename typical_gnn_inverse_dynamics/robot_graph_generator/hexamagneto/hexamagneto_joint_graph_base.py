import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from typical_gnn_inverse_dynamics.robot_graph_generator.hexamagneto.hexamagneto_graph_definition import HexaMagnetoGraphEdgeSender,HexaMagnetoGraphEdgeReceiver
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

######################################
#   DEFINE MAGNETO TOPOLOGY GRAPH    #
######################################
class HexaMagnetoJointGraphBase(RobotGraph):
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




