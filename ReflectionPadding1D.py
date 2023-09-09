import os
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec

'''
  1D Reflection Padding
  Attributes:
    - padding: (padding_left, padding_right) tuple
'''
class ReflectionPadding1D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0],
              input_shape[1] + self.padding[0] + self.padding[1],
              input_shape[2]
              )
        return shape

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return tf.pad(input_tensor,  [[0, 0], [padding_left, padding_right], [0, 0]], mode='SYMMETRIC')
        #return tf.pad(input_tensor,  [[0, 0], [padding_left, padding_right], [0, 0]], mode='REFLECT')
