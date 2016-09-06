
# coding: utf-8

import cPickle as cp
import numpy as np
import pydev
import nnet_tf


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = cp.load(file('cifar10.data'))

    net = nnet_tf.ConfigNetwork('net.conf', 'cifar10')

    net.fit(train_x, train_y)

    # simple_test
    x = train_x[:3000]
    y = train_y[:3000]
    pred_y = net.predict(x)
    print nnet_tf.precision_01(y, pred_y)


