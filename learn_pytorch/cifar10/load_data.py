#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 
import cPickle as cp
import numpy as np
import pydev
import sys
import random
import threading

cifar10_dir='/home/nickgu/lab/practice/dataset/cifar-10-batches-py/'
#cifar10_dir='/home/psdz/lab/practice/dataset/cifar-10-batches-py/'
#cifar10_dir='/home/gusimiu/lab/datasets/cifar10/cifar-10-batches-py/'
#cifar10_dir='/Users/gusimiu/lab/datasets/cifar10/cifar-10-batches-py/'


file_list = map(lambda x:cifar10_dir+x, [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5',
])

def load_files(filenames):
    xs = []
    ys =[]

    for filename in filenames:
        d = cp.load(file(filename))
        xs.append( d['data'].reshape(10000, 3, 32, 32) )
        ys += d['labels']

    x = np.concatenate( xs )
    y = np.array( ys )
    '''
    y = np.zeros((len(ys), 10))
    for idx, l in enumerate(ys):
        y[idx][l] = 1.
    print y
    '''

    print 'load data over'
    print 'x_shape: ', x.shape
    print 'y_shape: ', y.shape
    return x, y

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


