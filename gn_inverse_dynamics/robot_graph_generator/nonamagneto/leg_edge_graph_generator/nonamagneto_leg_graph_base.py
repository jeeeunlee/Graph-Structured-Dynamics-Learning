import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.robot_graph_generator.nonamagneto.leg_edge_graph_generator.nonamagneto_leg_definition import NonaMagnetoGraphEdgeSender,NonaMagnetoGraphEdgeReceiver
from gn_inverse_dynamics.robot_graph_generator.robot_graph import RobotGraph

######################################
#   DEFINE MAGNETO TOPOLOGY GRAPH    #
######################################
class NonaMagnetoLegGraphBase(RobotGraph):
    def __init__(self):
        receivers = []
        senders = []
        for sender in NonaMagnetoGraphEdgeSender:
            senders.append(NonaMagnetoGraphEdgeSender[sender]) 
        for receiver in NonaMagnetoGraphEdgeReceiver:
            receivers.append(NonaMagnetoGraphEdgeReceiver[receiver])         
        super().__init__(senders, receivers)
        
    # def generate_graph_dict(self, global_features, nodes, edges)
    # def concat_graph(self, graph_dicts)




