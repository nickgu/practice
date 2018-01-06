#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import dataset

def lr_net(x, y):
    w = tf.Variable(tf.random_normal(shape=[4, 3], stddev=0.01))
    b = tf.Variable(tf.random_normal(shape=[3], stddev=0.01))
    y_ = tf.matmul(x, w) + b

    lr_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
            )
    return lr_cost, y_

def nn_net(x, y):
    w_hidden = tf.Variable(tf.random_normal(shape=[4, 8], stddev=0.01))
    b_hidden = tf.Variable(tf.random_normal(shape=[8], stddev=0.01))
    hidden_ = tf.nn.relu( tf.matmul(x, w_hidden) + b_hidden )

    w = tf.Variable(tf.random_normal(shape=[8, 3], stddev=0.01))
    b = tf.Variable(tf.random_normal(shape=[3], stddev=0.01))
    nn_y_ = tf.matmul(hidden_, w) +b

    nn_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=nn_y_, labels=y)
            )
    return nn_cost, nn_y_

if __name__=='__main__':
    iris_data = dataset.IrisData()
    X, Y = iris_data.data()
    print X.shape
    print Y.shape

    x = tf.placeholder(tf.float32, shape=[None, 4])
    y = tf.placeholder(tf.float32, shape=[None, 3])

    cost, y_ = lr_net(x, y)
    #cost, y_ = nn_net(x, y)

    #train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost) # construct optimizer
    train_op = tf.train.AdamOptimizer(0.1).minimize(cost) 
    #train_op = tf.train.AdagradOptimizer(0.1).minimize(cost) 
    #train_op = tf.train.RMSPropOptimizer(0.01).minimize(cost) 
    #train_op = tf.train.FtrlOptimizer(0.6).minimize(cost) 

    predict_op = tf.argmax(y_, 1) # at predict time, evaluate the argmax of the logistic regression

    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for i in range(20):
            sess.run(train_op, feed_dict={x:X, y:Y})
            prec = np.mean(  np.argmax(Y, axis=1) == sess.run(predict_op, feed_dict={x:X}) )

            print 'iter=%d, prec=%.5f' % (i, prec)





