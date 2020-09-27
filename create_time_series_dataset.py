# -*- coding: utf-8 -*-
"""
create dataset
"""
import os
import numpy as np
import pandas as pd

def load_pop(fnm='pop_t'):
    mat = np.loadtxt(fnm)
    dt = mat[1,0] - mat[0,0]
    return dt, mat[:,1]

def createTimeSeries(pop, look_back, n_step_out, ind_0, ind_last, f_path):
    list_time_series = []
    for i in range(ind_0, ind_last-look_back-n_step_out):
        list_time_series.append(pop[i:i+look_back+n_step_out])
    # arr_time_series = np.array(list_time_series)
    df = pd.DataFrame(list_time_series)
    df.to_csv(f_path, index=False)
def createTimeSeriesFromFile(pop_fnm='pop_t',
         time_series_data_dir='train_time_series',
         fnm='1.csv',
         look_back=24,
         n_step_out=1,
         ind_0=0,
         ind_last=80):
    if not os.path.exists(time_series_data_dir):
        os.makedirs(time_series_data_dir)
    f_path = os.path.join(time_series_data_dir, fnm)
    dt, pop = load_pop()
    createTimeSeries(pop, look_back, n_step_out, ind_0, ind_last, f_path)
    return dt, pop
if __name__=='__main__':
    createTimeSeriesFromFile()