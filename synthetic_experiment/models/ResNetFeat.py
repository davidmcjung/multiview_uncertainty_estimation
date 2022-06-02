import tensorflow as tf
from .spectral_normalization import SpectralNormalization

class ResNetFeat(tf.keras.Model):
  """Defines a multi-layer residual network."""
  def __init__(self, num_layers=3, num_hidden=128, use_sn=False):
    super().__init__()

    self.num_hidden = num_hidden
    self.num_layers = num_layers

    # Defines the hidden layers.
    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
    self.dense_layers = [self.make_dense_layer(use_sn) for _ in range(num_layers)]

  def call(self, inputs):
    # Projects the 2d input data to high dimension.
    hidden = self.input_layer(inputs)

    # Computes the resnet hidden representations.
    for i in range(self.num_layers):
      resid = self.dense_layers[i](hidden)
      hidden += resid

    return hidden

  def make_dense_layer(self, use_sn):
    # Uses the Dense layer as the hidden layer.
    dense_layer = tf.keras.layers.Dense(self.num_hidden, activation="relu")
    if use_sn == False:
        return dense_layer
    else:
        return SpectralNormalization(dense_layer, norm_multiplier=.9)