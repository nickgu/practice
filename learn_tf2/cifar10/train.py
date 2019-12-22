#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

#from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import datetime

import cPickle as cp
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model, models, layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from lsuv_init import LSUVinit 

import models

def prepare_data():
    print('=== Load data ===')

    # prepare data.
    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Add a channels dimension
    #x_train = x_train[..., tf.newaxis]
    #x_test = x_test[..., tf.newaxis]

    #path = '/home/psdz/lab/practice/dataset/cifar-10-batches-py/'
    path = '/home/nickgu/lab/practice/dataset/cifar-10-batches-py/'
    #path='/Users/gusimiu/lab/practice/dataset/cifar-10-batches-py/'
    xs = []
    ys = []
    for i in range(5):
        d = cp.load(file(path + 'data_batch_%d' % (i+1)))
        xs.append( d['data'].reshape(10000, 3, 32, 32) )
        ys.append( d['labels'] )

    x_train = np.concatenate( xs )
    y_train = np.concatenate( ys )

    d = cp.load(file(path + 'test_batch'))
    x_test = d['data'].reshape(10000, 3, 32, 32)
    y_test = d['labels']

    # note here.
    x_train = np.transpose(x_train, (0,2,3,1))
    x_test = np.transpose(x_test, (0,2,3,1))

    print('=== Load data over ===')
    print('LenX=%d' % len(x_train))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


if __name__=='__main__':
    # Model, Loss, Optimizer, Data
    log_dir = 'models'

    batch_size = 64
    model_builder = models.V7_ResNet9
    model = model_builder()
    x_train, y_train, x_test, y_test = prepare_data()

    run_name = '%s_bs%d_%s' % (model_builder.__name__,
            batch_size,
            datetime.datetime.now().strftime("%Y%m%d:%H:%M:%S"))
    print '=== Run Name: %s ===' % run_name 
    logs = log_dir = 'logs/%s/%s' % (log_dir, run_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if len(sys.argv)>=2 and sys.argv[1] == 'load_model':
        print 'Load model from last save.'
        model.load_weights('./models/train_save')

    '''
    # learning rate scheduler.
    step = tf.Variable(0, trainable=False)
    boundaries = [50, 200,   800,    2000, 4000, 8000, 10000]
    values = [0.1, 0.01, 0.005, 0.0005, 0.0001, 5e-4, 1e-4, 1e-5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values)
    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)
    '''

    #model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    #model = LSUVinit(model,x_train[:batch_size,:,:,:]) 

    '''
    y_test = np.array(y_test)
    #train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    model.fit(x_train, y_train,
            epochs=150,
            validation_data=[x_test, y_test],
            batch_size=512,
            callbacks=[tensorboard_callback])
    '''

    # train translation.
    datagen = ImageDataGenerator(
            #featurewise_center=True,
            #featurewise_std_normalization=True,
            rotation_range=60,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True)
    datagen.fit(x_train)
    model.fit(
            #train_ds, 
            datagen.flow(x_train, y_train, batch_size=batch_size), 
            epochs=500, 
            validation_data=datagen.flow(x_test, y_test,batch_size=batch_size),
            callbacks=[tensorboard_callback])

    '''
    model.fit_generator(
            #train_ds,
            datagen.flow(x_train, y_train, batch_size=batch_size), 
            #steps_per_epoch = 50000 / batch_size, 
            epochs=500, 
            #workers=4,
            validation_data=datagen.flow(x_test, y_test,batch_size=batch_size),
            callbacks=[tensorboard_callback])
    '''

    model.summary()
    model.save_weights('./models/train_save')

