#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import dataset

if __name__=='__main__':
    iris_data = dataset.IrisData()
    X, Y = iris_data.data()
    print X.shape
    print Y.shape

    # make net
    x = tf.placeholder(tf.float32, shape=[None, 4])
    y = tf.placeholder(tf.float32, shape=[None, 3])

    w = tf.Variable(tf.random_normal(shape=[4, 3], stddev=0.01))
    b = tf.Variable(tf.random_normal(shape=[3], stddev=0.01))
    y_ = tf.matmul(x, w) + b

    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
            )

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct optimizer
    predict_op = tf.argmax(y_, 1) # at predict time, evaluate the argmax of the logistic regression

    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for i in range(10000):
            sess.run(train_op, feed_dict={x:X, y:Y})
            prec = np.mean(  np.argmax(Y, axis=1) == sess.run(predict_op, feed_dict={x:X}) )

            print 'iter=%d, prec=%.5f' % (i, prec)





