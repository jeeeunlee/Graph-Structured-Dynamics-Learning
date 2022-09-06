from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt


NUM_LAYERS_ = 2  # Hard-code number of layers in the edge/node/global models.
FFNN_LATENT_SIZE_ = 512 #24 #48 #16  # Hard-code latent layer sizes for demos.


def make_mlp_model_ffnn():
  """make_mlp_model_edge Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  print("make mlp model edge = LATENT_SIZE = {}".format(FFNN_LATENT_SIZE_))
  return snt.Sequential([
      snt.nets.MLP([FFNN_LATENT_SIZE_] * NUM_LAYERS_, activate_final=True), #, dropout_rate=0.2
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

def make_linear_model(output_size):
  return snt.Linear(output_size, name="output")

class MLPFeedForwardModel(snt.Module):
    """
    MLPFeedForwardModel
    """
    def __init__(self, 
                num_layer = NUM_LAYERS_,
                latent_size=FFNN_LATENT_SIZE_, 
                output_size = None,
                name="MLPFeedForwardModel"):
        super(MLPFeedForwardModel, self).__init__(name=name)
        # update latent size
        global NUM_LAYERS_
        global FFNN_LATENT_SIZE_
        NUM_LAYERS_ = num_layer
        FFNN_LATENT_SIZE_ = latent_size

        # 
        self._network = make_mlp_model_ffnn()
        self._output_transform = make_linear_model(output_size=output_size)

    def __call__(self, inputs):
        latent = self._network(inputs)        
        return self._output_transform(latent)