# Symmetry net

"""Model architectures for the demos in TensorFlow 2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

#default
LATENT_SIZE=64
NUM_LAYER=2
OUTPUT_SIZE=5

def make_mlp_model(latent_size=LATENT_SIZE, num_layer=NUM_LAYER, nameadd=''):
  """make_mlp_model Instantiates a new MLP, followed by LayerNorm.
  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  print("make mlp model >> LATENT_SIZE = {}".format(latent_size))
  return snt.Sequential([
      snt.nets.MLP([latent_size] * num_layer, activate_final=True, name='mlp'+nameadd), #, dropout_rate=0.2
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True, name='layer_norm'+nameadd)  ])

def make_linear_model(output_size):
  return snt.Linear(output_size, name="output")

class Aggregator():
  """ agg """
  def __init__(self,
              aggregator=tf.math.reduce_sum,
              repeater=tf.repeat):
    self._aggregator = aggregator
    self._repeater = repeater
  
  def __call__(self, inputs):
    axis_dim = tf.shape(inputs)
    res = self._aggregator(inputs, axis=1, keepdims=True)
    return self._repeater(res, repeats=axis_dim[1], axis=1)
    

class EncodeProcessDecode(snt.Module):
  """Full encode-process-decode model.
  """
  def __init__(self,
               latent_size=LATENT_SIZE,
               num_layer=NUM_LAYER,
               output_size=OUTPUT_SIZE,
               num_processing_steps=3,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)

    self._encoder = make_mlp_model(latent_size, num_layer,nameadd='_enc')
    self._core = make_mlp_model(latent_size,num_layer,nameadd='_core')
    self._decoder = make_mlp_model(latent_size,num_layer,nameadd='_dec')
    self._output_transform = make_linear_model(output_size=output_size)
    self._aggregator = Aggregator()
    self._num_processing_steps = num_processing_steps


  def __call__(self, input_op):
    latent0 = self._encoder(input_op)
    latent = latent0

    for _ in range(self._num_processing_steps):
      latent = self._aggregator(latent)
      core_input = tf.concat([latent0, latent], axis=2)
      latent = self._core(core_input)
      
    decoded_op = self._decoder(latent)
    return self._output_transform(decoded_op)
  