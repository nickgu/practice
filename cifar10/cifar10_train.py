
# coding: utf-8

import numpy as np
import pydev
import nnet_tf
import load_data

import sys

if __name__ == '__main__':
    model_path = sys.argv[1]

    train_x, train_y = load_data.load_one_part()
    print train_x.shape
    print train_y.shape

    net = nnet_tf.ConfigNetwork('net.conf', 'cifar10')
    net.fit(train_x, train_y)
    net.save(model_path)

