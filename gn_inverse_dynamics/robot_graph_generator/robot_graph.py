import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

class RobotGraph():
    # topology of robot is defined
    def __init__(self, senders, receivers):
        self.senders = senders
        self.receivers = receivers

    def generate_graph_dict(self, global_features, nodes, edges):        
        return {
            "globals": global_features,
            "nodes": nodes,
            "edges": edges,
            "receivers": self.receivers,
            "senders": self.senders
        }

    def concat_graph(self, graph_dicts):
        ## TODO : concat graph based on reciever and senders
        ''' 
        input: 
            graph_dicts : list of graph dict [graph_dict1, graph_dict2, ... ]
        output:
            graph_dict : concatenated graphs 
                        with the same topology along the features dimension
        '''
        global_feature = [] # u1 + u2 : [u1,u2]
        nodes = [] # V1:[v11, v12, ...] + V2:[v21, v22, ...] = V:[[v11,v21], ...]  
        edges = [] # E1:[e11, e12, ...] + E2:[e21, e22, ...] = E:[[e11,e21], ...]

        for graph in graph_dicts :
            # concatenate global
            if(type(graph['globals']) is list):
                global_feature.extend(graph['globals'])
            elif(graph['globals'] is None):
                pass
            else:
                global_feature.append(graph['globals'])

            # concatenate node
            if(len(nodes) == 0):
                nodes = [ x[:] for x in graph['nodes'] ] # init
            else:
                nodes_temp = (graph['nodes'][:])
                for node, node_temp in zip(nodes, nodes_temp):
                    node.extend(node_temp) # concat

            # concatenate edge
            if(len(edges) == 0):
                edges = [ x[:] for x in graph['edges'] ]# init
            else:
                edges_temp = (graph['edges'][:])
                for edge, edge_temp in zip(edges, edges_temp):
                    edge.extend(edge_temp) # concat

        return self.generate_graph_dict(global_feature, nodes, edges)

    def check_print(self, graph_dict):        
        # print(self.graph_dict)
        print("globals : ")
        self.pretty_print(graph_dict["globals"])
        print("nodes : ")
        self.pretty_print(graph_dict["nodes"])
        print("edges : ")
        self.pretty_print(graph_dict["edges"])
        print("receivers : ")
        self.pretty_print(graph_dict["receivers"])
        print("senders : ")
        self.pretty_print(graph_dict["senders"])

    def pretty_print(self, input_list):
        if(isinstance(input_list, list)):
            if(len(input_list)>0 and isinstance(input_list[0], list)):
                for i, element in enumerate(input_list):
                    print("{} : {}".format(i, element))
            else:
                print(input_list)

        else:
            print(input_list)
        
            