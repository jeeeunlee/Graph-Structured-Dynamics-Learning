import os 
import sys
import tensorflow as tf

CURRENT_DIR_PATH = os.getcwd()
sys.path.append(CURRENT_DIR_PATH)

from sn_inverse_dynamics.train_model import TorqueErrorModel

# from gn_inverse_dynamics.train_model.state_error_model import StateErrorModel
# from gn_inverse_dynamics.train_model.reward_weighted_model import RewardWeightedModel

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    # log directory
    parser.add_argument("--log_dir", type=str, default='')

    # graph-nets load/save
    parser.add_argument("--save_model_path", type=str, default="/saved_model")
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--load_model_path", type=str, default="savemodel/saved_model")

    # dataset path
    parser.add_argument("--dataset_path", type=str, default="a_dataset/tfData")
    
    # learning parameters
    parser.add_argument("--learning_rate", type=float, default=5e-4)

    parser.add_argument("--epoch_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--output_size", type=int, default=12)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--num_processing_steps", type=int, default=3)

    # training options
    # shuffle?
    parser.add_argument("--buffer_size", type=int, default=None)    
    # validation check?
    parser.add_argument("--validation_test", type=int, default=1)
    parser.add_argument("--test_dataset_path", type=str, default="a_dataset/tfData")

    parser.add_argument("--error_model", type=str, default="torque")

    

    args = parser.parse_args()

    if(args.error_model == "torque"):
        train_model = TorqueErrorModel(args)
    # elif(args.error_model == "state"):
    #     train_model = StateErrorModel(args)
    # elif(args.error_model == "reward"):
    #     train_model = RewardWeightedModel(args)
    else:
        exit()

    train_model.run()

