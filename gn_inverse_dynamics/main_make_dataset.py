import collections
import itertools
import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from gn_inverse_dynamics.utils.myutils import *
from gn_inverse_dynamics.robot_graph_generator.magneto.magneto_graph_generator import *

def make_dataset(args):
    # SET DATA PATH
    base_data_path = CURRENT_DIR_PATH + args.urdf_path
    raw_data_dir = CURRENT_DIR_PATH + args.raw_data_dir
    dataset_path = CURRENT_DIR_PATH + args.dataset_dir

    print("base_data_path = {}".format(base_data_path))
    print("raw_data_dir = {}".format(raw_data_dir))
    print("dataset_path = {}".format(dataset_path))    

    pass_param = PassThresholdParam(1000,5)

    data_generator = MagnetoLegGraphGenerator(
                        base_data_path = base_data_path,
                        traj_data_path = raw_data_dir,
                        pass_param = pass_param)

    try:
        data_generator.generate_tf_dataset(dataset_path = dataset_path,
                                        dataset_size = args.dataset_size,
                                        args = args )
    except ValueError:
        exit()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--urdf_path", type=str, default="/gn_inverse_dynamics/robot_graph_generator/magneto/magneto_simple.urdf")
    parser.add_argument("--raw_data_dir", type=str, default="/dataset/magneto/rawData")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/magneto/tfData")
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--split_data", type=bool, default=True)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--shuffle", type=bool, default=False)  

    args = parser.parse_args()
    make_dataset(args)