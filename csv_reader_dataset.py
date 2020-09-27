#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:47:13 2020

@author: xieyu
"""
import os
import tensorflow as tf


def preprocess(n_features=1, look_back=24, n_step_out=12):
    def func(line):
        # default values. 
        # if missing, set the missing one to be the default value.
        defs = [0.] * (look_back * n_features + n_step_out)
        fields = tf.io.decode_csv(line, record_defaults=defs)
        x = tf.stack(fields[:-n_step_out])
        x = tf.reshape(x, [look_back,n_features])
        y = tf.stack(fields[-n_step_out:])
        return x, y
    return func

def csv_reader_dataset(filepaths, repeat=100, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32,
                       n_features=13, look_back=48, n_step_out=12,
                       shuffle=True):
#    print(filepaths)
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    my_preprocess = preprocess(n_features=n_features,
                               look_back=look_back, 
                               n_step_out=n_step_out)
    dataset = dataset.map(my_preprocess, num_parallel_calls=n_parse_threads)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    else:
        dataset = dataset.repeat(repeat)
    return dataset.batch(batch_size).prefetch(1)

def selectCsvsFromDir(data_dir):
    fnms = os.listdir(data_dir)
    fnms = filter(lambda x: x.endswith('.csv'), fnms) # csv files
    filepaths = [os.path.join(data_dir, fnm) for fnm in fnms]
    #print(filepaths)
    return filepaths

