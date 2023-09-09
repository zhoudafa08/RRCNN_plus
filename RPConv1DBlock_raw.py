import os
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.regularizers import l2
from ReflectionPadding1D import ReflectionPadding1D
from tensorflow.keras.layers import Input

class RPConv1DBlock_raw(Layer):
    def __init__(self, half_kernel_size=7, **kwargs):
        self.half_kernel_size = half_kernel_size
        self.RP = ReflectionPadding1D(padding=(self.half_kernel_size, 
                      self.half_kernel_size))
        self.C1D_act = Conv1D(1, 2*self.half_kernel_size+1, 
                      activation='tanh', 
                      use_bias=True,
                      kernel_regularizer=l2(1e-4)
                      )
        super(RPConv1DBlock_raw, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel1 = self.add_weight(name='kernel1',
                        shape=(2*self.half_kernel_size+1, 1, 1),
                        initializer='uniform',
                        #regularizer=keras.regularizers.l1(1e-4),
                        trainable=True)
        super(RPConv1DBlock_raw, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0],
              input_shape[1],
              input_shape[2]
              )
        return shape
    
    def call(self, inputs):
        x = self.RP(inputs)
        x = self.C1D_act(x)
        x = self.RP(x)
        filter1 = self.kernel1 ** 2
        filter1 = tf.scalar_mul(tf.math.reciprocal(tf.reduce_sum(filter1)), filter1)
        x = tf.nn.conv1d(x, filter1, stride=1, padding='VALID')
        x = tf.keras.layers.Subtract()([inputs, x])
        return x
    
    def get_config(self):
        return {"half_kernel_size":self.half_kernel_size}
