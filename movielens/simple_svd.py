#! /bin/env python
# encoding=utf-8
# author: nickgu
#
#   Usage:
#       python simple_svd.py <movielens_user_data> [seperater=<tab>]
#       
#   example:
#       python simple_svd.py ml-100k/u.data
#
#      movielens_user_data format:
#
#      <uid>[tab]<iid>[tab]<rating>[tab]timestamp
#   
#

import sys
import random

import numpy as np
import sklearn.decomposition as dc

# refer to simple_fm.py in this dir.
import simple_fm

class SVDLearner:
    def __init__(self):
        self.__svd = dc.TruncatedSVD(n_components = 64)

    def fit(self, train):
        max_uid = max(map(lambda x:x[0], train))
        max_iid = max(map(lambda x:x[1], train))
        print max_uid, max_iid
        X = np.ndarray( (max_uid+1, max_iid+1) )
        X.fill(0)
        for uid, iid, rating in train:
            X[uid][iid] = rating

        # make X = U * V
        self.U = self.__svd.fit_transform(X)
        # svd does not support interface for accessing V
        # so we need to make a calculation.
        self.V = np.mat(self.U).I * X
        self.X_ = self.U * self.V
        
        print 'train over.'

    def predict(self, uid, iid):
        # use default value
        # inv ratio = 74%
        #return 4

        # use random select
        # inv ratio = 45% (almost)
        #return random.randint(1, 5)

        # use SVD decomposition:
        # inv ratio = 32.09%
        if uid >= self.X_.shape[0]: return 4
        if iid >= self.X_.shape[1]: return 4

        reg = self.X_[uid, iid]
        return reg

def load_data(filename, sep='\t'):
    data = []
    # load data.
    for line in file(filename).readlines():
        uid, iid, rating, timestamp = line.strip().split(sep)
        if uid == 'userId':
            continue

        uid = int(uid)
        iid = int(iid)
        rating = float(rating)
        data.append( (uid, iid, rating) )

    # make 19/20 as train, 1/20 as test.
    split = len(data) // 20 * 19
    train, test = data[:split], data[split:]
    return train, test


def inverse_ratio(order):
    # in y_, -rating ascendant order.
    order = sorted(order, key=lambda x:(x[0], -x[1]))

    prev_count = {}
    inverse_count = 0
    total_count = 0

    for i, (_, r) in enumerate(order):
        for key, count in prev_count.iteritems():
            total_count += count
            if key > r:
                inverse_count += count

        # update previous count 
        prev_count[r] = prev_count.get(r, 0) + 1

    inv_ratio = inverse_count * 100.  / total_count
    return inv_ratio, inverse_count, total_count


if __name__=='__main__':
    filename = sys.argv[1]
    path = './'

    if len(sys.argv)>2:
        path = sys.argv[2]
        print >> sys.stderr, 'path=[%s]' % path
        train, test = load_data(filename)
    else:
        train, test = load_data(filename)

    print 'load data over train=%d test=%d' % (len(train), len(test))

    # TEST: sklearn SVD: 
    #  minimum 32% inverse ratio.
    #learner = SVDLearner()
    #learner.fit(train)

    # TEST: pyfm, inverse_ratio:
    # factor=10, use_info=False
    #   e20  : 19.05%
    #   e100 : 19.05%
    # factor=10, use_info=True
    #   e100 : 18.22%
    # factor=16, use_info=True
    #   e20  : 19.17%
    #   e100 : 18.63% 
    #learner = simple_fm.SimpleFMLearner(iter=140, factor=6, use_info=True, path=path)
    #learner.fit(train)

    # TEST: svdfeature
    # use svdfeature and make user-rated_item as feature, item_rated_user as feature.
    # learning_rate=0.0001, reg: wd_user=wd_item=0.003
    #   e100 : 17.33%
    #   e200 : 17.11%
    #   e300 : 17.02% 
    #   e450 : 16.95%
    # svd_feature inplement
    # EXTERNAL CODE.

    # TEST fastFM.
    # it's very very fast..
    # no use_info(only UID, MID):
    #   e30: 17.9% (ALS, iter=30, rank=2)
    # hard to tune the sgd.
    #   epoch 1000(1000*10w) with 0.0005 step_size(learning_rate): 17.34
    #
    # NOTICE: fastFM seems not to support centOS.   
    #
    '''
    from fastFM import als,mcmc,sgd
    learner = simple_fm.SimpleFMLearner(
            #external_fm = sgd.FMRegression(n_iter=100000000, step_size=0.0005, init_stdev=0.1, rank=12),
            external_fm = als.FMRegression(n_iter=30, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5),
            #external_fm = mcmc.FMRegression(n_iter=1000, init_stdev=0.1, rank=4),
            use_info = True,
            path = path,
            )
    learner.fit(train)
    '''

    # TEST tffm.
    # https://github.com/geffy/tffm
    from tffm import TFFMRegressor
    import tensorflow as tf
    learner = simple_fm.SimpleFMLearner(
            external_fm = TFFMRegressor(        
                order=2, 
                rank=12, 
                optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
                n_epochs=300, 
                batch_size=128,
                init_std=0.001,
                reg=0.001,
                input_type='sparse'
            ),
            use_info = True,
            path = path,
            )
    learner.fit(train)

    # calculate inverse count.
    order = []
    for uid, iid, rating in test:
        y_ = learner.predict(uid, iid)
        #print >> sys.stderr, '%d\t%d\t%.4f\t%d' % (uid, iid, y_, rating)
        order.append( (y_, rating) )

    inv_ratio, inverse_count, total_count = inverse_ratio(order)

    print 'inverse_ratio: %.2f%% (%d/%d)' % (
            inv_ratio, inverse_count, total_count)





