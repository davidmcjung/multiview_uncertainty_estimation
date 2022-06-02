import tensorflow as tf
from .ResNetFeat import ResNetFeat

class Baseline(tf.keras.Model):
  def __init__(self, C, resnet_layers=3, resnet_hidden=128):
      super().__init__()
      self.C = C
      self.experts = []
      self.model = tf.keras.models.Sequential()
      self.model.add(tf.keras.layers.Input((2)))
      self.model.add(ResNetFeat(resnet_layers, resnet_hidden))
      self.model.add(tf.keras.layers.Flatten())
      self.model.add(tf.keras.layers.Dense(C, activation=None))
      self.loss =  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  def call(self, X, y, training, concat=False):
      if concat:
          X = tf.concat([X[i] for i in range(len(X))], axis=1)
      logits = self.model(X, training=training)
      loss = self.loss(y, logits)
      loss = tf.math.reduce_mean(loss)
      return logits, loss