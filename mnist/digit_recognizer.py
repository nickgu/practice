#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import pydev
import csv
import sys
import sklearn
import logging
import numpy
from sklearn import preprocessing

if __name__=='__main__':
    train_file = file('data/train.csv', 'r')
    test_file = file('data/test.csv','r')
    validation = False

    output_file = sys.stdout
    if len(sys.argv)==2:
        output_filename = sys.argv[1]
        print >> sys.stderr, 'outputfile = %s' % output_filename
        output_file = file(output_filename, 'w')

    train_reader = csv.reader(train_file)
    train_X = []
    train_Y = []
    for row in train_reader:
        if row[0] == 'label':
            continue

        # init m
        m = numpy.ndarray(10)
        for i in range(10): m[i] = 0

        y = int(row[0])
        m[y] = 1.
        x = numpy.array( map(float, row[1:]) )
        x /= 256.
        train_X.append(x)
        train_Y.append(m)

    test_reader = csv.reader(test_file)
    test_X = []
    for row in test_reader:
        if row[0] == 'pixel0':
            continue

        x = numpy.array( map(float, row) )
        x /= 256.
        test_X.append(x)
    pydev.err('read over train=%d,%d test=%d' % (len(train_X), len(train_Y), len(test_X)))

    from sklearn import ensemble
    import nnet_tf
    #learner = ensemble.(n_estimators=100)
    #learner = ensemble.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=0)
    learner = nnet_tf.ConfigNetwork('conf/net.conf', 'mnist_conv2d_big', output_01=True)

    # validation train and predict.
    if validation:
        v_x = train_X[:len(train_X)/5*4]
        v_y = train_Y[:len(train_Y)/5*4]
        t_x = train_X[len(train_X)/5*4:]
        t_y = train_Y[len(train_X)/5*4:]
        learner.fit(v_x, v_y)
        tty = learner.predict(t_x)
        pred = filter(lambda x:x==0, map(lambda x:x.mean(), tty - t_y) )
        print '%.3f' % (len(pred) * 1. / len(tty))

    # formal train and predict.
    else:
        learner.fit(train_X, train_Y)
        ans = learner.predict(test_X)

        # pring header.
        print >> output_file, 'ImageId,Label'
        for idx, y in enumerate(ans):
            d = 0
            for u, v in enumerate(y):
                if v>.5: d=u
            print >> output_file, '%d,%d' % (idx+1,d)



