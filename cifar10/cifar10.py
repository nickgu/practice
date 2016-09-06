
# coding: utf-8

import numpy as np
import pydev
import nnet_tf
import load_data

if __name__ == '__main__':
    train_x, train_y = load_data.load_data()
    test_x, test_y = load_data.load_data()

    print train_x.shape, train_x[0][:20]
    print train_y.shape, train_y[:20]
    print test_x.shape, test_x[0][:20]
    print test_y.shape, test_y[:20]

    net = nnet_tf.ConfigNetwork('net.conf', 'cifar10')

    net.fit(train_x, train_y)

    pred_y = net.predict(test_x)
    print nnet_tf.precision_01(test_y, pred_y)


