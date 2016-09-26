
# coding: utf-8

import numpy as np
import pydev
import nnet_tf
import load_data

import sys

if __name__ == '__main__':
    model_path = sys.argv[1]

    net = nnet_tf.ConfigNetwork('net.conf', 'cifar10')
    net.load(model_path)

    train_x, train_y = load_data.load_one_part()
    print train_x.shape
    print train_y.shape

    pred_y = net.predict(test_x)
    print nnet_tf.precision_01(test_y, pred_y)


