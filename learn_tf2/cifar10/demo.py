#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

#from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import cPickle as cp
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model, models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(16, 3, activation='relu')
    self.pool1 = MaxPool2D()
    self.conv2 = Conv2D(32, 3, activation='relu')

    self.flatten = Flatten()
    self.d1 = Dense(256, activation='relu')
    self.d2 = Dense(256, activation='relu')
    self.d3 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)

    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__=='__main__':
    # Model, Loss, Optimizer, Data

    print('=== Load data ===')

    # prepare data.
    #mnist = tf.keras.datasets.mnist

    path = '/home/nickgu/lab/practice/dataset/cifar-10-batches-py/'

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

    print x_train.shape
    print y_train.shape


    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('=== Load data over ===')
    print('LenX=%d' % len(x_train))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    #x_train = x_train[..., tf.newaxis]
    #x_test = x_test[..., tf.newaxis]

    # train translation.
    '''
    datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    datagen.fit(x_train)
    '''

    # prepare train and test.
    '''
    train_ds = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    '''

    '''
    train_ds = tf.data.Dataset.from_generator(
                    datagen.flow, args=[x_train, y_train],
                    output_types=(tf.float32, tf.float32)
                    ).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_generator(
                    datagen.flow, args=[x_test, y_test],
                    output_types=(tf.float32, tf.float32)
                    ).batch(32)
    '''


    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=20, 
            validation_data=(x_test, y_test))

    '''
    # make Model.
    model = MyModel()

    # make Loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # make Optimizer
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    print('=== Ready to train! ===')
    EPOCHS = 100
    for epoch in range(EPOCHS):
        print('Step1 ...')
        idx = 0
        for images, labels in train_ds:
            sys.stdout.write('%c%d' %(13, idx))
            idx += 1

            train_step(images, labels)
        sys.stdout.write('\n')

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                             train_loss.result(),
                             train_accuracy.result()*100,
                             test_loss.result(),
                             test_accuracy.result()*100))



    

    '''
