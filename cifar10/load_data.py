#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 
import cPickle as cp
import numpy as np
import pydev
import sys

#cifar10_dir='/home/nickgu/lab/datasets/cifar10/cifar-10-batches-py/'
cifar10_dir='/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/'

file_list = map(lambda x:cifar10_dir+x, [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5',
])

def load_one_part():
    train_x = []
    train_y = []
    m = cp.load(file(file_list[0]))
    for x in m['data']:
        train_x.append( pydev.zip_channel(x, 3) )

    train_x = np.array(train_x) / 255.
    train_y = m['labels']
    train_y = pydev.index_to_one_hot(np.array(train_y), 10)
    return train_x, train_y

def load_data():
    ''' load all data. '''
    train_x = []
    train_y = []
    for fn in file_list:
        m = cp.load(file(fn))
        for x in m['data']:
            train_x.append( pydev.zip_channel(x, 3) )
        train_y += m['labels']
        
    train_x = np.array(train_x) / 255.
    train_y = pydev.index_to_one_hot(np.array(train_y), 10)
    return train_x, train_y

def load_test():
    m = cp.load(file(cifar10_dir + 'test_batch'))
    test_x = []
    for x in m['data']:
        test_x.append( pydev.zip_channel(x, 3) )

    test_x = np.array(test_x) / 255.
    test_y = pydev.index_to_one_hot(np.array(m['labels']), 10)
    return test_x, test_y


if __name__=='__main__':
    train_x = []
    train_y = []
    for fn in file_list:
        m = cp.load(file(fn))
        for x in m['data']:
            train_x.append( pydev.zip_channel(x, 3) )
        train_y += m['labels']
        
    train_x = np.array(train_x)  #/ 255.
    train_y = pydev.index_to_one_hot(np.array(train_y), 10)

    m = cp.load(file('/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/test_batch'))
    test_x = []
    for x in m['data']:
        test_x.append( pydev.zip_channel(x, 3) )

    test_x = np.array(test_x) #/ 255.
    test_y = pydev.index_to_one_hot(np.array(m['labels']), 10)

    print >> sys.stderr, 'load over, begin to dump.'
    cp.dump([train_x, train_y, test_x, test_y], file('cifar10.data', 'w'))

