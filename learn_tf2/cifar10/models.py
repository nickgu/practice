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
    # v4 deep model, fetch 87% acc
    model = models.Sequential()
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


