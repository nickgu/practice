
# coding: utf-8

import numpy as np
import pydev
import nnet_tf
import load_data

import tensorflow as tf
import sys


class Tester:
    def __init__(self):
        self.__best_precision = 0
        self.__best_iteration = 0
        
        self.sess = tf.Session()
        self.s_test_precision = tf.placeholder(tf.float32)
        a = tf.summary.scalar('validation/precision/test', self.s_test_precision)
        self.merged = tf.summary.merge([a])

    def test(self, predict, summary_writer=None, iteration=0):
        pred_y = net.predict(test_x)
        precision = nnet_tf.precision_01(
                    pydev.index_to_one_hot(test_y, 10), 
                    pred_y
                )[0]

        if summary_writer:
            _, summary = self.sess.run([self.s_test_precision, self.merged], 
                    feed_dict = {
                        self.s_test_precision: precision,
                        })
            summary_writer.add_summary(summary, iteration)

        if precision > self.__best_precision:
            self.__best_precision = precision
            self.__best_iteration = iteration
        print >> sys.stderr, '[iter=%d] Test precision: %.3f (best=%.3f, at iteration %d)' % (
                iteration, precision,
                self.__best_precision,
                self.__best_iteration)

class Preprocessor:
    def __init__(self):
        pass

    def distorted(self, queue_x, queue_y):
        CropSize = 24
        distorted_image = tf.cast(queue_x, tf.float32)
        distorted_image = tf.random_crop(distorted_image, [CropSize, CropSize, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        distorted_image = tf.image.per_image_standardization(distorted_image)

        return distorted_image, queue_y

if __name__ == '__main__':
    model_path = sys.argv[1]
    net = nnet_tf.ConfigNetwork('net.conf', 'cifar10_tf_tutor')

    temp_data = pydev.TempStorage('normal_data', 'temp/normal_data.ts')
    if temp_data.has_data():
        train_x = temp_data.read()
        train_y = temp_data.read()
        test_x = temp_data.read()
        test_y = temp_data.read()

    else:
        train_x, train_y = load_data.load_all_data()
        test_x, test_y = load_data.load_test()

        temp_data.write(train_x)
        temp_data.write(train_y)
        temp_data.write(test_x)
        temp_data.write(test_y)

    # process test_X
    sess = tf.Session()
    CropSize = 24
    image_in = tf.placeholder(tf.uint8, shape=[32, 32, 3])
    image = tf.cast(image_in, tf.float32)

    resize_image = tf.image.resize_image_with_crop_or_pad(image, CropSize, CropSize)
    test_image = tf.image.per_image_standardization(resize_image)
    
    processed_X = []
    for image in test_x:
        image_out = sess.run( test_image, feed_dict={image_in:image})
        processed_X.append( image_out )

    test_x = processed_X

    tester = Tester()
    pre = Preprocessor()
    net.fit(train_x, train_y, tester.test, callback_iteration=400, preprocessor=pre.distorted)
    print >> sys.stderr, 'Final prediction'
    tester.test(net.predict)
    net.save(model_path)
    




