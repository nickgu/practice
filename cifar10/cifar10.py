
# coding: utf-8

import numpy as np
import pydev
import nnet_tf
import load_data

if __name__ == '__main__':
    train_x, train_y = load_data.load_one_part()

    print train_x.shape
    print train_y.shape

    net = nnet_tf.ConfigNetwork('net.conf', 'cifar10')

    net.fit(train_x, train_y)

    # simple_test
    x = train_x
    y = train_y
    pred_y = net.predict(x)
    print nnet_tf.precision_01(y, pred_y)


