import collections
import itertools
import os 
import sys

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)



from typical_gnn_inverse_dynamics.utils.myutils import *
from typical_gnn_inverse_dynamics.robot_graph_generator.magneto.magneto_graph_generator import *

def make_dataset(args):
    # SET DATA PATH
    urdf_data_path = CURRENT_DIR_PATH + args.urdf_path
    raw_data_dir = CURRENT_DIR_PATH + args.raw_data_dir
    dataset_path = CURRENT_DIR_PATH + args.dataset_dir

    print("urdf_data_path = {}".format(urdf_data_path))
    print("raw_data_dir = {}".format(raw_data_dir))
    print("dataset_path = {}".format(dataset_path))    

    pass_param = PassThresholdParam(100,5)

    data_generator = MagnetoGraphGenerator(
                        urdf_data_path = urdf_data_path,
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

    parser.add_argument("--urdf_path", type=str, default="/typical_gnn_inverse_dynamics/robot_graph_generator/magneto/magneto_2_floatingbase.urdf")
    parser.add_argument("--raw_data_dir", type=str, default="/a_dataset/rawData/ros_pnc/caseall")
    parser.add_argument("--dataset_dir", type=str, default="/a_dataset/tfData/ros_pnc/gnn/caseall")
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--split_data", type=bool, default=True)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--shuffle", type=bool, default=True)  

    args = parser.parse_args()
    make_dataset(args)