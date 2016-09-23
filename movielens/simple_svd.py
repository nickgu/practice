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

    # make 4/5 as train, 1/5 as test.
    split = len(data) // 5 * 4
    train, test = data[:split], data[split:]
    return train, test

if __name__=='__main__':
    filename = sys.argv[1]
    if len(sys.argv)>2:
        print 'sep=[%s]' % sys.argv[2]
        train, test = load_data(filename, sep=sys.argv[2])
    else:
        train, test = load_data(filename)

    print 'load data over train=%d test=%d' % (len(train), len(test))

    # sklearn SVD: 
    #  minimum 32% inverse ratio.
    #learner = SVDLearner()

    # pyfm:
    # inverse_ratio : 19.05% at 20 epoch.
    # inverse_ratio : 19.05% at 100 epoch.
    learner = simple_fm.SimpleFMLearner()
    
    learner.fit(train)

    # calculate inverse count.
    order = []
    for uid, iid, rating in test:
        y_ = learner.predict(uid, iid)
        #print >> sys.stderr, '%d\t%d\t%.4f\t%d' % (uid, iid, y_, rating)
        order.append( (y_, rating) )


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
    print 'inverse_ratio: %.2f%% (%d/%d)' % (
            inv_ratio, inverse_count, total_count)





