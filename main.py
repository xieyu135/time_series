#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:29:33 2020

@author: xieyu
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from create_time_series_dataset import createTimeSeriesFromFile
from lstm_model import LstmModel
from roll_forward import rollForward

def plotPopIni(pop, ind_sep):
    plt.plot(pop)
    plt.plot(pop[:ind_sep], 'bo')
    plt.show()
def main(pop_fnm='pop_t',
         time_series_train_dir='train_time_series',
         time_series_test_dir='test_time_series',
         n_features=1,
         look_back=24,
         n_step_out=1,
         epochs=400,
         ind_sep=80,
         create_train_time_series_on=1,
         train_on=1,
         load_lstm_model_on=0):
    if create_train_time_series_on:
        dt, pop = createTimeSeriesFromFile(pop_fnm=pop_fnm,
                 time_series_data_dir=time_series_train_dir,
                 fnm='1.csv',
                 look_back=look_back,
                 n_step_out=n_step_out,
                 ind_0=0,
                 ind_last=ind_sep)
    m = LstmModel(n_features=n_features, 
                 look_back=look_back, 
                 n_step_out=n_step_out,
                 n_layer=2,
                 n_neuron=50,
                 activation = 'tanh',
                 epochs = epochs,
                 batch_size = 10,
                 loss='mean_squared_error', 
                 optimizer='adam',
                 verbose=1,
                 train_data_dir="train_time_series",
                 test_data_dir="test_time_series",
                 test_result_dir='test_result',
                 model_path='lstm_model.h5')   
    if train_on:
        m.build()
        m.train()
    if load_lstm_model_on:
        m.load_model()
    #dataset_x = pop[ind_sep-look_back:ind_sep]
    #dataset_x = tf.reshape(dataset_x, [1,look_back,n_features])
    #y_predict = m.model.predict(dataset_x)
    #print(y_predict)
    #print(pop[ind_sep])
    # plt.plot(pop)
    # plotPopIni(pop,ind_sep)
    pop_predict = rollForward(pop, 
        m,
        look_back=look_back, 
        n_step_out=n_step_out, 
        ind_sep=ind_sep, 
        n_features=n_features)
    plt.plot(pop)
    plt.plot(np.arange(ind_sep,pop.shape[0]),pop_predict[ind_sep:], "bo")
    plt.show()

if __name__=='__main__':
    main(pop_fnm='pop_t',
         time_series_train_dir='train_time_series',
         time_series_test_dir='test_time_series',
         look_back=60,
         n_step_out=1,
         epochs=60,
         ind_sep=200,
         create_train_time_series_on=1,
         train_on=0,
         load_lstm_model_on=1)
    
    
