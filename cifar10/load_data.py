#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 
import cPickle as cp
import numpy as np
import pydev
import sys

if __name__=='__main__':
    file_list = [
        '/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/data_batch_1',
        '/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/data_batch_2',
        '/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/data_batch_3',
        '/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/data_batch_4',
        '/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/data_batch_5',
    ]


    train_x = []
    train_y = []
    for fn in file_list:
        m = cp.load(file(fn))
        for x in m['data']:
            train_x.append( pydev.zip_channel(x, 3) )
        train_y += m['labels']
        
    train_x = np.array(train_x)
    train_y = pydev.index_to_one_hot(np.array(train_y), 10)

    m = cp.load(file('/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/test_batch'))
    test_x = []
    for x in m['data']:
        test_x.append( pydev.zip_channel(x, 3) )

    test_y = pydev.index_to_one_hot(np.array(m['labels']), 10)

    print >> sys.stderr, 'load over, begin to dump.'
    cp.dump([train_x, train_y, test_x, test_y], file('cifar10.data', 'w'))

