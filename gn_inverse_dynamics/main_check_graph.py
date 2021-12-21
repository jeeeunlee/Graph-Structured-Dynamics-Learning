import collections
import itertools
import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.utils import myutils as myutils
from gn_inverse_dynamics.robot_graph_generator.magneto.magneto_graph_generator import *

def check_dataset(args):
    # SET DATA PATH
    base_data_path = CURRENT_DIR_PATH + args.urdf_path
    raw_data_dir = CURRENT_DIR_PATH + args.raw_data_dir
    save_data_dir = CURRENT_DIR_PATH + args.save_data_dir
 
    pass_param = PassThresholdParam(1000,5)

    data_generator = MagnetoLegGraphGenerator(
                        base_data_path = base_data_path,
                        traj_data_path = raw_data_dir,
                        pass_param = pass_param)

    gen = data_generator.gen_graph_dicts()
    myutils.create_folder(save_data_dir)
    edge1 = myutils.Logger(save_data_dir + '/edge1.csv')
    edge2 = myutils.Logger(save_data_dir + '/edge2.csv')
    edge3 = myutils.Logger(save_data_dir + '/edge3.csv')
    edge4 = myutils.Logger(save_data_dir + '/edge4.csv')
    global0 = myutils.Logger(save_data_dir + '/global.csv')

    for _ in range(100):
        next_graph = next(gen)
        edge1.record_list(next_graph[0]["edges"][0])
        edge2.record_list(next_graph[0]["edges"][1])
        edge3.record_list(next_graph[0]["edges"][2])
        edge4.record_list(next_graph[0]["edges"][3])
        global0.record_list(next_graph[0]["globals"])
        # print("next_graph = ")
        # print(next_graph[0]["edges"][0])
        # print(next_graph[0]["edges"][1])
        # print(next_graph[0]["edges"][2])
        # print(next_graph[0]["edges"][3])
        # print(next_graph[0]["globals"])

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--urdf_path", type=str, default="/gn_inverse_dynamics/robot_graph_generator/magneto/magneto_simple.urdf")
    parser.add_argument("--raw_data_dir", type=str, default="/dataset/magneto/rawData210809")
    parser.add_argument("--save_data_dir", type=str, default="/dataset/magneto/datacheck")

    args = parser.parse_args()
    check_dataset(args)