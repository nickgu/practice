
# coding: utf-8

import numpy as np
import pydev
import nnet_tf
import load_data

import sys

if __name__ == '__main__':
    model_path = sys.argv[1]
    net = nnet_tf.ConfigNetwork('net.conf', 'cifar10_32')

    train_x, train_y = load_data.load_all_data(dup=4, image_preprocess=True, shuffle=True)
    #train_x, train_y = load_data.load_data()

    #test_x, test_y = train_x, train_y
    #test_x, test_y = load_data.load_one_part(dup=1, image_preprocess=True)
    test_x, test_y = load_data.load_test()


    '''
    temp_store = pydev.TempStorage('one5_one', 'temp/one5_one.ts')
    if temp_store.has_data():
        train_x, train_y, test_x, test_y = temp_store.read()

    else:
        train_x, train_y = load_data.load_one_part(dup=5, image_preprocess=True, shuffle=True)
        #train_x, train_y = load_data.load_data()

        #test_x, test_y = train_x, train_y
        test_x, test_y = load_data.load_one_part(dup=1, image_preprocess=True)
        #test_x, test_y = load_data.load_test()

        temp_store.write([train_x, train_y, test_x, test_y])
    '''

    def tester(predict):
        pred_y = net.predict(train_x[:10000])
        train_precision = nnet_tf.precision_01(train_y[:10000], pred_y)[0]

        pred_y = net.predict(test_x)
        precision = nnet_tf.precision_01(test_y, pred_y)[0]
        print >> sys.stderr, 'Test precision: %.3f (train_P:%.3f)' % (
                precision, train_precision)


    net.fit(train_x, train_y, tester)
    net.save(model_path)

