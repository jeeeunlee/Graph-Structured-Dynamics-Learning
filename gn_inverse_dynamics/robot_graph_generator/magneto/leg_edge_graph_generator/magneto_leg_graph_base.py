import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.robot_graph_generator.magneto.leg_edge_graph_generator.magneto_leg_definition import MagnetoGraphEdgeSender,MagnetoGraphEdgeReceiver
from gn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

######################################
#   DEFINE MAGNETO TOPOLOGY GRAPH    #
######################################
class MagnetoLegGraphBase(RobotGraph):
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




