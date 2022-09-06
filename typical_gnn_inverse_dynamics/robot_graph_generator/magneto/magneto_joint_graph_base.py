import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_graph_definition import MagnetoGraphEdgeSender,MagnetoGraphEdgeReceiver
from typical_gnn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

######################################
#   DEFINE MAGNETO TOPOLOGY GRAPH    #
######################################
class MagnetoJointGraphBase(RobotGraph):
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




