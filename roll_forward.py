#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:19:29 2020

@author: xieyu
"""
import tensorflow as tf
import numpy as np

def rollOneStep(dataset_x, y_predict):
    n_step_out = y_predict.shape[0]
    if n_step_out>0:
        dataset_x = tf.Variable(dataset_x)
        dataset_x[:,:-n_step_out].assign(dataset_x[:,n_step_out:])
        dataset_x[:,-n_step_out:].assign(y_predict)
        dataset_x = tf.convert_to_tensor(dataset_x)
    return dataset_x
def rollForward(pop, 
        lstm_obj,
        look_back=24, 
        n_step_out=1, 
        ind_sep=80, 
        n_features=1):
    n_all = pop.shape[0]
    n_times_predict = (n_all - ind_sep)//n_step_out
    ind_tmp_0 = ind_sep - look_back
    ind_tmp_end = ind_sep
    dataset_x = pop[ind_tmp_0:ind_tmp_end]
    dataset_x = tf.reshape(dataset_x,[1,look_back,n_features])
    # print(dataset_x.shape)
    y_predict = np.array([])
    pop_predict = pop.copy()
    for i in range(n_times_predict):
        y_predict = lstm_obj.model.predict(dataset_x)
        ind_tmp_predict_0 = ind_sep+i*n_step_out
        ind_tmp_predict_end = ind_tmp_predict_0 + n_step_out
        pop_predict[ind_tmp_predict_0:ind_tmp_predict_end] = y_predict[0]
        # print(y_predict)
        y_predict = tf.convert_to_tensor(y_predict, dtype=tf.float64)
        y_predict = tf.reshape(y_predict, [1,n_step_out,n_features])
        dataset_x = rollOneStep(dataset_x, y_predict)
    return pop_predict
