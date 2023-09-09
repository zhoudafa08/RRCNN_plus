import os
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.regularizers import l2
from ReflectionPadding1D import ReflectionPadding1D
from tensorflow.keras.layers import Input

class RPConv1DBlock_attention(Layer):
    def __init__(self, half_kernel_size=7, **kwargs):
        self.half_kernel_size = half_kernel_size
        self.half_kernel_size_hf = int(half_kernel_size/2)
        self.half_kernel_size_hhf = int(half_kernel_size/4)
        self.RP = ReflectionPadding1D(padding=(self.half_kernel_size, 
                      self.half_kernel_size))
        self.RP_hf = ReflectionPadding1D(padding=(self.half_kernel_size_hf, 
                      self.half_kernel_size_hf))
        self.RP_hhf = ReflectionPadding1D(padding=(self.half_kernel_size_hhf, 
                      self.half_kernel_size_hhf))
        self.C1D_act = Conv1D(1, 2*self.half_kernel_size+1, 
                      activation='relu', 
                      use_bias=True,
                      kernel_regularizer=l2(1e-4)
                      )
        self.C1D_act_hf = Conv1D(1, 2*self.half_kernel_size_hf+1, 
                      activation='relu', 
                      use_bias=True,
                      kernel_regularizer=l2(1e-4)
                      )
        self.C1D_act_hhf = Conv1D(1, 2*self.half_kernel_size_hhf+1, 
                      activation='relu', 
                      use_bias=True,
                      kernel_regularizer=l2(1e-4)
                      )
        self.PointConv1D_relu = Conv1D(1, 1,
                           activation='relu',
                           kernel_regularizer=l2(1e-4),
                           use_bias=True)
        self.PointConv1D_sigmoid = Conv1D(1, 1,
                           kernel_regularizer=l2(1e-4),
                           activation='sigmoid',
                           use_bias=True)
        super(RPConv1DBlock_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel1 = self.add_weight(name='kernel1',
                        shape=(2*self.half_kernel_size+1, 1, 1),
                        initializer='uniform',
                        regularizer=tf.keras.regularizers.l2(1e-4),
                        trainable=True)
        self.kernel2 = self.add_weight(name='kernel2',
                        shape=(2*self.half_kernel_size+1, 5, 1),
                        initializer='uniform',
                        regularizer=tf.keras.regularizers.l1(1e-4),
                        trainable=True)
        self.kernel3 = self.add_weight(name='kernel3',
                        shape=(2*self.half_kernel_size+1, 1, 1),
                        initializer='uniform',
                        regularizer=tf.keras.regularizers.l1(1e-4),
                        trainable=True)
        self.pool_size = input_shape[1] - (2*self.half_kernel_size+1) +1
        self.AvgPool1D = AveragePooling1D(pool_size=self.pool_size, strides=1, padding='valid')
        super(RPConv1DBlock_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0],
              input_shape[1],
              input_shape[2]
              )
        return shape
    
    def call(self, inputs):
        #A = self.AvgPool1D(inputs)
        #A = self.PointConv1D_relu(A)
        #A= BatchNormalization(axis=-1, 
        #     center=True, scale=True, 
        #     beta_initializer='ones', 
        #     gamma_initializer='ones',
        #     moving_mean_initializer='ones',
        #     moving_variance_initializer='ones')(A)
        #A = tf.keras.layers.Dropout(0.5)(A)
        #A = self.PointConv1D_relu(A)
        ##A = self.PointConv1D_sigmoid(A)
        #A= BatchNormalization(axis=-1, 
        #     center=True, scale=True, 
        #     beta_initializer='ones', 
        #     gamma_initializer='ones',
        #     moving_mean_initializer='ones',
        #     moving_variance_initializer='ones')(A)
        #A = tf.keras.layers.Dropout(0.5)(A)
        
        x = self.RP(inputs)
        #filter1 = self.kernel1
        #filter11 = tf.multiply(A, tf.squeeze(filter1,axis=[-1]))
        #xx = tf.transpose(tf.expand_dims(x, -1), perm=[3, 1, 2,0])
        #filter11 = tf.transpose(tf.expand_dims(filter11, -1), perm=[1, 2, 0, 3])
        #xx = tf.nn.depthwise_conv2d(xx, filter11, strides=[1, 1, 1, 1],
        #               padding='VALID')
        #xx = tf.keras.layers.Dropout(0.5)(xx)
        #xx = tf.squeeze(tf.transpose(xx, perm=[3, 1, 2, 0]), axis=[-1])
        xx = tf.keras.layers.Attention()([inputs, inputs])
        x_f = self.C1D_act(x) 
        x_f = tf.keras.layers.Dropout(0.5)(x_f)
        x_hf = self.C1D_act_hf(x) 
        x_hf = tf.keras.layers.Dropout(0.5)(x_hf)
        x_hhf = self.C1D_act_hhf(x) 
        x_hhf = tf.keras.layers.Dropout(0.5)(x_hhf)
        x = tf.concat([self.RP(inputs), self.RP(x_f), self.RP_hf(x_hf), self.RP_hhf(x_hhf),  self.RP(xx)], -1)
        #x = tf.concat([self.RP(inputs), self.RP(x_f), self.RP_hf(x_hf), self.RP_hhf(x_hhf)], -1)
        filter2 = self.kernel2
        x = tf.nn.conv1d(x, filter2, stride=1, padding='VALID')
        x = tf.nn.tanh(x)
        x = self.RP(x)
        
        filter3 = self.kernel3 ** 2
        filter3 = tf.scalar_mul(tf.math.reciprocal(tf.reduce_sum(filter3)), filter3)
        #filter2 = tf.nn.softmax(self.kernel2)
        #x = tf.concat([x, self.RP(inputs)], -1)
        x = tf.nn.conv1d(x, filter3, stride=1, padding='VALID')
        x = tf.keras.layers.Subtract()([inputs, x])
        return x
    
    def get_config(self):
        return {"half_kernel_size":self.half_kernel_size}
