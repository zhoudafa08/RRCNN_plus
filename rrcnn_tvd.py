import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.layers import Conv1D, Input, Activation
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
from ExtMidSig import ExtMidSig
from CustomTV_MSE import CustomTV_MSE
from ReflectionPadding1D import ReflectionPadding1D
from RPConv1DBlock_raw import RPConv1DBlock_raw
import numpy as np
import pandas as pd
import sys,pywt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import scipy.signal as ss
import time
from TVD_IC import TVD_IC

tf.config.run_functions_eagerly(True)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Generate data
def data():
    # add Gaussian white noise
    def wgn(x, snr):
        snr = 10**(snr/10.0)
        xpower = np.sum(x**2)/len(x)
        npower = xpower / snr
        noise = np.random.randn(len(x)) * np.sqrt(npower)
        return noise
    length = 2400
    snr = 25
    exd_length = int(0.5*length)
    mode = 'symmetric'
    x_train = np.empty([length+2*exd_length, 1])
    y_train = np.empty([2*length, 1])
    t = np.linspace(0, 6, length)
    for j in range(5, 15):
        x1 = np.cos(j *np.pi * t)
        x2 = np.cos((j+1.5)*np.pi*t)
        tmp_x = pywt.pad(x1 + x2, exd_length, mode)
        tmp_noise = wgn(tmp_x.reshape(-1), snr)
        x_train = np.c_[x_train, tmp_x+tmp_noise]
        y_train = np.c_[y_train, np.r_[x2, x1]]
        tmp_x = pywt.pad(x2, exd_length, mode)
        tmp_noise = wgn(tmp_x.reshape(-1), snr)
        x_train = np.c_[x_train, tmp_x+tmp_noise]
        y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
        x2 = np.cos((j+1.5)*np.pi*t + t * t + np.cos(t))
        tmp_x = pywt.pad(x1 + x2, exd_length, mode)
        tmp_noise = wgn(tmp_x.reshape(-1), snr)
        x_train = np.c_[x_train, tmp_x+tmp_noise]
        y_train = np.c_[y_train, np.r_[x2, x1]]
        tmp_x = pywt.pad(x2, exd_length, mode)
        tmp_noise = wgn(tmp_x.reshape(-1), snr)
        x_train = np.c_[x_train, tmp_x+tmp_noise]
        y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
        for i in range(2, 20):
            x2 = np.cos(j*i*np.pi*t)
            tmp_x = pywt.pad(x1 + x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, x1]]
            tmp_x = pywt.pad(x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
            x2 = np.cos(j*i*np.pi*t + t * t + np.cos(t))
            tmp_x = pywt.pad(x1 + x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, x1]]
            tmp_x = pywt.pad(x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
    

    print(x_train.shape, y_train.shape)  
    x_train = np.delete(x_train, 0, axis=1)
    y_train = np.delete(y_train, 0, axis=1)
    indices = np.arange(x_train.shape[1])
    np.random.seed(0)
    np.random.shuffle(indices)
    x_sample = x_train[:, indices]
    y_sample = y_train[:, indices]
    train_num = int(0.7*x_sample.shape[1])
    x_train = x_sample[:, :train_num]
    x_test = x_sample[:, train_num:]
    y_train = y_sample[:, :train_num]
    y_test = y_sample[:, train_num:]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    x_train = x_train.transpose().reshape(-1, length+2*exd_length, 1)
    y_train = y_train.transpose().reshape(-1, 2*length, 1)
    x_test = x_test.transpose().reshape(-1, length+2*exd_length, 1)
    y_test = y_test.transpose().reshape(-1, 2*length, 1)
    print(x_train.shape, y_train.shape)
    return x_train, y_train, x_test, y_test

min_error = 10.0
# create model
def create_model(x_train, y_train, x_test, y_test):
    #Layer_num = {{choice(range(5, 16))}}
    inputs = Input(shape=x_train.shape[1:], dtype='float32')
    ## cell 1
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(inputs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(5, 100, 5))}})(outs)
    
    ## cell 1
    inputs2 = inputs - outs
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(inputs2)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = TVD_IC(0.04, 4)(outs)
    outs1 = ExtMidSig(x_train.shape[1], int(y_train.shape[1]/2))(outs)
    
    ## cell 2
    inputs3 = inputs2 - outs
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(inputs3)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock_raw({{choice(range(10, 150, 5))}})(outs)
    outs = TVD_IC(0.04, 4)(outs)
    outs2 = ExtMidSig(x_train.shape[1], int(y_train.shape[1]/2))(outs)
    
    final_outs = tf.keras.layers.Concatenate(axis=1)([outs1, outs2]) 
    model = Model(inputs=inputs, outputs=final_outs)
    model.summary()
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss=CustomTV_MSE(0.01), metrics=['mse'])
    
    # train
    model.fit(x=x_train, y=y_train, epochs={{choice(range(20,150,10))}}, 
                batch_size=16, verbose=1, validation_data=(x_test, y_test))
    
    pred = model.predict(x_test)
    mse = mean_squared_error(pred[:,:,0], y_test[:, :,0])
    length = pred.shape[1]
    pred1 = pred[:, :int(length/2), :]
    pred2 = pred[:, int(length/2):, :]
    tv_pred11 = tf.math.reduce_mean(tf.abs(pred1[:,:-1,:]-pred1[:,1:,:]))
    tv_pred21 = tf.math.reduce_mean(tf.abs(pred2[:,:-1,:]-pred2[:,1:,:]))
    tv_pred13 = tf.math.reduce_mean(tf.abs(pred1[:,:-3,:]-3*pred1[:,1:-2,:]+3*pred1[:,2:-1,:]-pred1[:,3:,:]))
    tv_pred23 = tf.math.reduce_mean(tf.abs(pred2[:,:-3,:]-3*pred2[:,1:-2,:]+3*pred2[:,2:-1,:]-pred2[:,3:,:]))
    score = mse+(tv_pred11+2*tv_pred13+tv_pred21+2*tv_pred23)*0.01
    weight_path = './models/model_dataset2_rrcnntvd.h5'
    #try:
    #    with open('rrcnn_manually.txt') as f:
    #        #min_error = float(f.read().strip().split('(')[1].split(',')[0])
    #        min_error = float(f.read().strip())
    #except FileNotFoundError:
    #    min_error = score
    if score <= min_error:
        model.save(weight_path)
        min_error = score
    #    with open('rrcnn_manually.txt','w') as f:
    #        f.write(str(min_error))
    #sys.stdout.flush()
    return {'loss': score, 'model': model, 'status': STATUS_OK}

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data()
    best_run, best_model = optim.minimize(model=create_model,
                            data=data, algo=tpe.suggest,
                            max_evals=5, trials=Trials())
    
    weight_path = './models/model_dataset2_rrcnntvd.h5'
    #best_model.save(weight_path)
    best_model = load_model(weight_path, \
            custom_objects={'ReflectionPadding1D':ReflectionPadding1D, 'RPConv1DBlock_raw':RPConv1DBlock_raw, 'ExtMidSig':ExtMidSig, 'CustomTV_MSE':CustomTV_MSE, 'TVD_IC':TVD_IC}) 

    pred = best_model.predict(x_train)
    rmse = np.sqrt(mean_squared_error(pred[:,:,0], y_train[:, :,0]))
    mae = mean_absolute_error(pred[:,:,0], y_train[:, :,0])
    mape = mean_absolute_percentage_error(pred[:,:,0], y_train[:, :,0])
    tv1 = np.mean(np.sum(np.abs((pred[:,1:,0]-pred[:,0:-1,0])), axis=1))
    tv2 = np.mean(np.sum(np.abs((pred[:,2:,0]-2*pred[:,1:-1,0]+pred[:,:-2,0])),axis=1))
    tv3 = np.mean(np.sum(np.abs((pred[:,3:,0]-3*pred[:,2:-1,0]+3*pred[:,1:-2,0]-pred[:,:-3,0])),axis=1))
    print('training performance:', mae, rmse, mape, tv1, tv2, tv3)
    tv1 = np.mean(np.sum(np.abs((y_train[:,1:,0]-  y_train[:,0:-1,0])), axis=1))
    tv2 = np.mean(np.sum(np.abs((y_train[:,2:,0]-2*y_train[:,1:-1,0]+  y_train[:,:-2,0])),axis=1))
    tv3 = np.mean(np.sum(np.abs((y_train[:,3:,0]-3*y_train[:,2:-1,0]+3*y_train[:,1:-2,0]-y_train[:,:-3,0])),axis=1))
    print('training ground truth performance:', tv1, tv2, tv3)
    
    pred = best_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(pred[:,:,0], y_test[:, :,0]))
    mae = mean_absolute_error(pred[:,:,0], y_test[:, :,0])
    mape = mean_absolute_percentage_error(pred[:,:,0], y_test[:, :,0])
    tv1 = np.mean(np.sum(np.abs((pred[:,1:,0]-pred[:,0:-1,0])), axis=1))
    tv2 = np.mean(np.sum(np.abs((pred[:,2:,0]-2*pred[:,1:-1,0]+pred[:,:-2,0])),axis=1))
    tv3 = np.mean(np.sum(np.abs((pred[:,3:,0]-3*pred[:,2:-1,0]+3*pred[:,1:-2,0]-pred[:,:-3,0])),axis=1))
    print('testing performance:', mae, rmse, mape, tv1, tv2, tv3)
    tv1 = np.mean(np.sum(np.abs((y_test[:,1:,0]-  y_test[:,0:-1,0])), axis=1))
    tv2 = np.mean(np.sum(np.abs((y_test[:,2:,0]-2*y_test[:,1:-1,0]+  y_test[:,:-2,0])),axis=1))
    tv3 = np.mean(np.sum(np.abs((y_test[:,3:,0]-3*y_test[:,2:-1,0]+3*y_test[:,1:-2,0]-y_test[:,:-3,0])),axis=1))
    print('testing ground truth performance:', tv1, tv2, tv3)
    
    
    # add Gaussian white noise
    def wgn(x, snr):
        snr = 10**(snr/10.0)
        xpower = np.sum(x**2)/len(x)
        npower = xpower / snr
        noise = np.random.randn(len(x)) * np.sqrt(npower)
        return noise
    
    length = 2400
    exd_length = int(0.5*length)
    t = np.linspace(0, 6, length)
    x1 = np.cos(5*np.pi*t)
    x2 = np.cos(8*np.pi*t+2*t*t+np.cos(t))
    #x2 = np.cos(7*np.pi*t)
    signal = pywt.pad(x1+x2, exd_length, 'symmetric')
    noise = wgn(signal.reshape(-1), 15)
    print(signal.shape)
    
    #inputs = (x1+x2).reshape(-1)
    inputs = (signal+noise).reshape(-1)
    layer_model = Model(inputs=best_model.input, outputs=best_model.layers[-1].output)
    start = time.time()
    pred = layer_model.predict(signal.reshape(1,length+2*exd_length,1))
    end = time.time()
    print('Time:', end-start)
    
