#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import tensorflow as tf

def k_max_pooling(input, k_max):
    topk_idx = tf.reverse(tf.nn.top_k(tf.nn.top_k(input, k_max).indices, k_max).values, [False, True])
    my_range = tf.expand_dims(tf.range(0, topk_idx.get_shape()[0]), 1)
    my_range_repeated = tf.tile(my_range, [1, k_max])
    full_indices = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(topk_idx, 2)])
    return tf.gather_nd(input, full_indices)

if __name__=='__main__':
    pass    
