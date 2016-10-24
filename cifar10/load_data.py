#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 
import cPickle as cp
import numpy as np
import pydev
import sys
import tensorflow as tf
import random
import threading

cifar10_dir='/home/nickgu/lab/datasets/cifar10/cifar-10-batches-py/'
#cifar10_dir='/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/'


file_list = map(lambda x:cifar10_dir+x, [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5',
])

def load_files(filenames):
    data_x = []
    data_y = []

    for filename in filenames:
        m = cp.load(file(filename))
        for idx, x in enumerate(m['data']):
            image = pydev.zip_channel(x, 3)
            image = image.reshape( (32, 32, 3) )
            data_x.append(image)
            data_y.append(m['labels'][idx])

            sys.stderr.write('%c%d image(s) loaded.' % (13, len(data_x)))
    sys.stderr.write('\nLoad image over.\n')

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y

def load_one_part():
    return load_files(file_list[:1])

def load_all_data():
    ''' load all data '''
    return load_files(file_list)

def load_test():
    ''' load test data '''
    return load_files([ cifar10_dir + 'test_batch' ])

if __name__=='__main__':
    pass


