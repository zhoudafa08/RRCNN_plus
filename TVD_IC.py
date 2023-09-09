import os
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.regularizers import l2
from ReflectionPadding1D import ReflectionPadding1D
from tensorflow.keras.layers import Input

class TVD_IC(Layer):
    def __init__(self, lam=1.0, Num=2, **kwargs):
        self.lam = lam
        self.Num = Num
        super(TVD_IC, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0],
              input_shape[1],
              input_shape[2]
              )
        return shape
    
    def call(self, inputs):
        xx = tf.squeeze(inputs, axis=-1)
        z = 0.0 * xx
        z = z[:, 0:-1]
        for i in range(self.Num): 
            y = tf.concat([tf.expand_dims(-z[:,0],axis=-1), z[:,0:-1]-z[:,1:]], 1)
            y = tf.concat([y, tf.expand_dims(z[:,-1],axis=-1)], 1)
            x = xx - y
            z = z + 1/3*(x[:,1:]-x[:,0:-1])
            z = tf.math.maximum(tf.math.minimum(z, self.lam/2), -self.lam/2)
        return tf.expand_dims(x, axis=-1)
    
    def get_config(self):
        return {"lam":self.lam, "Num":self.Num}
