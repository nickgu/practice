#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import datetime

import cPickle as cp
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model, models, layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from lsuv_init import LSUVinit 


def V4_model():
    # v4 deep model, fetch 87% acc (batch=64)
    # VGG-like construction.
    model = models.Sequential()
    # test step

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def V5_model():
    # v5 deep model, refer to david model.
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(48, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(48, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(80, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(80, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(80, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(80, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(80, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

class ConvBn(tf.keras.layers.Layer):
    def __init__(self, out_channel, kernel_size=3, strides=1, padding='same'):
        super(ConvBn, self).__init__()
        self.__seq = models.Sequential()
        self.__seq.add(layers.Conv2D(out_channel, (kernel_size, kernel_size), padding=padding, strides=strides))
        self.__seq.add(layers.BatchNormalization(axis=3))
        self.__seq.add(layers.ReLU())

    def call(self, x):
        return self.__seq(x)

class ResConvBn(tf.keras.layers.Layer):
    def __init__(self, stack_count, out_channel):
        super(ResConvBn, self).__init__()
        self.__seq = models.Sequential()
        for i in range(stack_count):
            self.__seq.add(ConvBn(out_channel, kernel_size=3))

    def call(self, x):
        return x + self.__seq(x)

def V6_ResNet9():
    # according to Top1 in DAWNBenchmark:
    #   https://github.com/wbaek/torchskeleton/releases/tag/v0.2.1_dawnbench_cifar10_release
    # validation accuracy 89.6%(adam, batchsize=64)
    model = models.Sequential()
    model.add(ConvBn(64))
    model.add(ConvBn(128, kernel_size=5, strides=2))
    model.add(ResConvBn(2, 128))
    model.add(ConvBn(256, kernel_size=3, strides=1))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(ResConvBn(2, 256))
    model.add(ConvBn(128, kernel_size=3, strides=1, padding='valid'))
    #model.add(layers.AdaptiveMaxPooling2D(1, 1)) # adaptive pooling from Pytorch.
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def V7_ResNet9():
    # according to Top1 in DAWNBenchmark:
    #   https://github.com/wbaek/torchskeleton/releases/tag/v0.2.1_dawnbench_cifar10_release
    # validation accuracy 89.6%(adam, batchsize=64)
    model = models.Sequential()
    model.add(ConvBn(64))
    model.add(ConvBn(128, kernel_size=5, strides=2))
    model.add(ResConvBn(2, 128))
    model.add(ConvBn(256, kernel_size=3, strides=1))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(ResConvBn(2, 256))
    model.add(ConvBn(128, kernel_size=3, strides=1, padding='valid'))
    #model.add(layers.AdaptiveMaxPooling2D(1, 1)) # adaptive pooling from Pytorch.
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    return model


    
class ConvResLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_shape, out_channel, shortcut_translation=False):
        super(ConvResLayer, self).__init__()
        self.__seq = models.Sequential()
        self.__seq.add(layers.Conv2D(out_channel, kernel_shape, padding='same'))
        self.__seq.add(layers.BatchNormalization(axis=3))
        self.__seq.add(layers.ReLU())
        self.__seq.add(layers.Conv2D(out_channel, kernel_shape, padding='same'))
        self.__seq.add(layers.BatchNormalization())

        if shortcut_translation:
            self.shortcut=models.Sequential()
            self.shortcut.add(layers.Conv2D(out_channel, kernel_shape, padding='same'))
            self.shortcut.add(layers.Dropout(0.35))
        else:
            self.shortcut=layers.Dropout(0.2)

    def call(self, x):
        y = self.__seq(x)
        if self.shortcut:
            x = self.shortcut(x)
        return x + y

def BuildResNet_34():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (7, 7), strides=2, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(ConvResLayer((3, 3), 64, True))
    model.add(ConvResLayer((3, 3), 64))
    model.add(ConvResLayer((3, 3), 64))

    model.add(ConvResLayer((3, 3), 128, True))
    model.add(ConvResLayer((3, 3), 128))
    model.add(ConvResLayer((3, 3), 128))

    model.add(ConvResLayer((3, 3), 256, True))
    model.add(ConvResLayer((3, 3), 256))

    model.add(ConvResLayer((3, 3), 512, True))
    model.add(ConvResLayer((3, 3), 512))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


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

