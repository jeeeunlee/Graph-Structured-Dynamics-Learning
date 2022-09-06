import os 
import sys
import pickle

CURRENT_DIR_PATH = os.getcwd()
print(CURRENT_DIR_PATH)
sys.path.append(CURRENT_DIR_PATH)

dataset_path = os.path.join( CURRENT_DIR_PATH, "a_dataset/rawData/hexa_magneto/case_all/walkset_0_0" )

f_q1 = open(dataset_path + "/q_sen.txt")
f_q2 = open(dataset_path + "/q_sen.txt")

next(f_q2)

from gn_inverse_dynamics.utils import myutils as myutils
for _ in range(10):
    print('----------------------')
    print(myutils.string_to_list(next(f_q1)) )
    print(myutils.string_to_list(next(f_q2)) )



# import tensorflow as tf
# import sonnet as snt
# from sn_inverse_dynamics import sn_model as myModels

# with open(dataset_path + '/element_spec', 'rb') as in_:
#     dataset_element_spec = pickle.load(in_)
# dataset_path = os.path.join( dataset_path, 'testdataset' )
# dataset = tf.data.experimental.load(dataset_path, element_spec=dataset_element_spec)


# Batch data check
# def gen():
#     for i in range(100):        
#         yield datafunc(i)

# def datafunc(i=0):
#     input_tf = tf.constant([2*i+1, 2*i+2, 2*i+3, 2*i+4, 2*i+5, 2*i+6, 2*i+7, 2*i+8], shape=[4,2],dtype=tf.float64)
#     output_tf = tf.constant([5*i+1, 5*i+2, 5*i+3, 5*i+4], shape=[4,1],dtype=tf.float64)
#     return input_tf, output_tf

# input_tf, output_tf= datafunc()

# input_spec = tf.TensorSpec.from_tensor(input_tf) #TensorSpec(shape=(4, 2), dtype=tf.int32, name=None)
# output_spec= tf.TensorSpec.from_tensor(output_tf) #TensorSpec(shape=(4, 1), dtype=tf.int32, name=None)

# dataset = tf.data.Dataset.from_generator(gen,output_signature=(input_spec,output_spec) )


# n_batch = 16
# model = myModels.EncodeProcessDecode(output_size=3)
# datasetbatch = dataset.batch(n_batch, drop_remainder=True)

# for databatch in datasetbatch:
#     print(tf.TensorSpec.from_tensor(databatch[0])) 
#     print(tf.TensorSpec.from_tensor(databatch[1])) 
#     res = model(databatch[0])
#     print(tf.TensorSpec.from_tensor(res)) 
#     break


# for databatch in datasetbatch:
#     # print(tf.TensorSpec.from_tensor(databatch[0])) #shape=(16, 4, 2)
#     # print(tf.TensorSpec.from_tensor(databatch[1])) #shape=(16, 4, 1)
   
#     res1 = databatch[0]
#     datashape = tf.shape(res1)
#     #print(res1) 
#     print(tf.TensorSpec.from_tensor(res1)) # shape=(16, 4, 2)

#     res2=tf.reduce_mean(databatch[0], axis=1, keepdims=True)
#     # print(res2) 
#     print(tf.TensorSpec.from_tensor(res2)) # shape=(16, 1, 2)

#     res3 = tf.repeat( res2, repeats=datashape[1], axis=1 )
#     #print(res3) 
#     print(tf.TensorSpec.from_tensor(res3)) # shape=(16, 4, 2)

#     res4 = tf.concat([res1, res3], axis=2)
#     #print(res4)
#     print(tf.TensorSpec.from_tensor(res4))
#     break
