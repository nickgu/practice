#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 
import cPickle as cp
import numpy as np
import pydev
import sys
import tensorflow as tf
import random

cifar10_dir='/home/nickgu/lab/datasets/cifar10/cifar-10-batches-py/'
#cifar10_dir='/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/'


file_list = map(lambda x:cifar10_dir+x, [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5',
])

def load_files(filenames, image_preprocess=True, dup=1, shuffle=False):
    data_x = []
    data_y = []

    #CropSize = 24
    CropSize = 32

    sess = tf.Session()
    image_in = tf.placeholder(tf.uint8, shape=[32, 32, 3])

    distorted_image = image_in
    distorted_image = tf.random_crop(distorted_image, [CropSize, CropSize, 3])

    if image_preprocess:
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.3)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.3, upper=1.7)
        #distorted_image = tf.image.per_image_whitening(distorted_image)

    for filename in filenames:
        m = cp.load(file(filename))
        for idx, x in enumerate(m['data']):
            image = pydev.zip_channel(x, 3)

            image = image.reshape( (32, 32, 3) )
            for i in range(dup):
                image_out = sess.run( distorted_image, feed_dict={image_in:image})
                image_out = image_out.reshape( [CropSize * CropSize * 3] )
                data_x.append(image_out)
                data_y.append(m['labels'][idx])

            sys.stderr.write('%c%d image(s) loaded.' % (13, len(data_x)))
    sys.stderr.write('\nLoad image over.\n')

    '''
    if image_preprocess:
        summary_writer = tf.train.SummaryWriter("./image_summary")

        images = np.array(data_x[:1000]).reshape([-1, 24, 24, 3])
        image_holder = tf.placeholder(tf.uint8, shape=(1000, 24, 24, 3))
        image_summary = tf.image_summary('distorted_image', image_holder, max_images=20)
        summary = sess.run(image_summary, feed_dict={image_holder:images})
        summary_writer.add_summary(summary, 0)
    '''

    if shuffle:
        print >> sys.stderr, 'Shuffle data..'
        data = zip(data_x, data_y)
        random.shuffle(data)
        data_x = map(lambda x:x[0], data)
        data_y = map(lambda x:x[1], data)

    # centrialize. (-1, 1)
    data_x = np.array(data_x) / 128. - 1.
    data_y = pydev.index_to_one_hot(np.array(data_y), 10)
    
    return data_x, data_y

def load_one_part(image_preprocess=True, dup=10, shuffle=False):
    return load_files(file_list[:1], image_preprocess=image_preprocess, dup=dup, shuffle=shuffle)

def load_all_data(image_preprocess=True, dup=10, shuffle=False):
    ''' load all data '''
    return load_files(file_list, image_preprocess=image_preprocess, dup=dup, shuffle=shuffle)

def load_test():
    return load_files([ cifar10_dir + 'test_batch' ], image_preprocess=True, dup=1, shuffle=False)

if __name__=='__main__':
    train_x = []
    train_y = []
    for fn in file_list:
        m = cp.load(file(fn))
        for x in m['data']:
            train_x.append( pydev.zip_channel(x, 3) )
        train_y += m['labels']
        
    train_x = np.array(train_x)  #/ 255.
    train_y = pydev.index_to_one_hot(np.array(train_y), 10)

    m = cp.load(file('/Users/nickgu/lab/datasets/cifar10/cifar-10-batches-py/test_batch'))
    test_x = []
    for x in m['data']:
        test_x.append( pydev.zip_channel(x, 3) )

    test_x = np.array(test_x) #/ 255.
    test_y = pydev.index_to_one_hot(np.array(m['labels']), 10)

    print >> sys.stderr, 'load over, begin to dump.'
    cp.dump([train_x, train_y, test_x, test_y], file('cifar10.data', 'w'))

