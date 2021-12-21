import pickle


def print_data_size(file_name):
    with open(file_name, 'rb') as in_:
            data_size = pickle.load(in_)
            print(data_size)

datafoldername = "/home/jelee/a_python_ws/graph-nets-physics/magneto-tf2-legnode/dataset/magneto/tfData_slope_combination/case1/"

print_data_size(datafoldername + "data_size")
print_data_size(datafoldername + "train_data_size")
print_data_size(datafoldername + "test_data_size")
print_data_size(datafoldername + "val_data_size")