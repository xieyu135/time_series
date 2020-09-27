#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:51:06 2020

@author: xieyu
"""
import os
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

from csv_reader_dataset import csv_reader_dataset
from csv_reader_dataset import selectCsvsFromDir


class LstmModel():
    def __init__(self,
                 n_features=1, 
                 look_back=24, 
                 n_step_out=1,
                 n_layer=2,
                 n_neuron=50,
                 activation = 'tanh',
                 epochs = 100,
                 batch_size = 32,
                 loss='mean_squared_error', 
                 optimizer='adam',
                 verbose=1,
                 train_data_dir="train_time_series",
                 test_data_dir="test_time_series",
                 test_result_dir='test_result',
                 model_path='lstm_model.h5'):
        self.n_features = n_features
        self.look_back = look_back
        self.n_step_out = n_step_out
        self.n_layer = n_layer
        self.n_neuron = n_neuron
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.test_result_dir = test_result_dir
        self.model_path = model_path
        self.model = None
        
    def build(self):
        self.model = Sequential()
        for i in range(self.n_layer):  # 构建多层网络
            if self.n_layer == 1:
                self.model.add(LSTM(self.n_neuron, 
                                    activation=self.activation, 
                                    input_shape=(None, self.n_features)))
            else:
                if i==0:
                    self.model.add(LSTM(self.n_neuron, 
                                        activation=self.activation, 
                                        return_sequences=True, 
                                        input_shape=(None, self.n_features)))
                elif i==self.n_layer-1:
                    self.model.add(LSTM(self.n_neuron,  
                                        activation=self.activation))
                else:
                    self.model.add(LSTM(self.n_neuron,  
                                        activation=self.activation,
                                        return_sequences=True))
        self.model.add(Dense(self.n_step_out))
        # Summary the structure of neural network/网络结构
        self.model.summary()
        # Compile the neural network/编译网络
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        
    def train(self):
        train_filepaths = selectCsvsFromDir(self.train_data_dir)
        dataset = csv_reader_dataset(train_filepaths,
                                     batch_size=self.batch_size,
                                     repeat=1,
                                     n_features=self.n_features,
                                     look_back=self.look_back,
                                     n_step_out=self.n_step_out)
#        test_filepaths = selectCsvs(os.listdir(self.test_data_dir))
        self.model.fit(dataset, epochs=self.epochs, 
                       verbose=self.verbose, shuffle=True)
        self.model.save(self.model_path)
        
    def test(self, fnm_y_err='y_err.dat'):
        test_filepaths = selectCsvsFromDir(self.test_data_dir)
        if not os.path.exists(self.test_result_dir):
            os.makedirs(self.test_result_dir)
        y_err = []
        for filepath in test_filepaths:
            fnm = os.path.split(filepath)[-1]
            test_result_filepath = os.path.join(self.test_result_dir, fnm)
            tmp_filepaths = [filepath]
            dataset = csv_reader_dataset(tmp_filepaths,
                                     repeat=1,
                                     n_features=self.n_features,
                                     look_back=self.look_back,
                                     n_step_out=self.n_step_out,
                                     shuffle=False)
            dataset_x = dataset.map(lambda X,y: X)
            dataset_y = dataset.map(lambda X,y: y)
            y_predict = self.model.predict(dataset_x)
            #print('type(y_predict):', type(y_predict))
            y_predict = pd.DataFrame(y_predict)
            y_predict.to_csv(test_result_filepath, index=False)
            
            y_predict = y_predict.values
            y_true = dataset_y.as_numpy_iterator()
            y_true = np.concatenate([arr_i for arr_i in y_true], axis=0)
            y_err_tmp = y_predict - y_true
            y_err.append(y_err_tmp)
        y_err = np.concatenate(y_err, axis=0)
        print(y_err.shape)
        np.savetxt(fnm_y_err, y_err)
        
    def load_model(self):
        self.model = load_model(self.model_path)