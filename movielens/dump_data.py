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
from sklearn.feature_extraction import DictVectorizer

from simple_fm import Info


def load_data(path):
    filename = path + 'u.data'
    sep = '\t'

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

    info = Info(path)
    X = []
    Y = []
    for userid, itemid, rating in data:
        d = { "user_id": str(userid), "movie_id": str(itemid)}
        d = info.process(userid, itemid, d)

        X.append(d)
        Y.append( rating )

    v = DictVectorizer()
    X = v.fit_transform(X)

    print >> sys.stderr, 'load data over train=%d test=%d' % (X.shape[0], len(Y))
    return X, Y


if __name__ == '__main__':
    X, Y = load_data(path=sys.argv[1])

    train_file = file('data/train.txt', 'w')
    test_file = file('data/test.txt', 'w')
    for ins_idx, y in enumerate(Y):
        line = '%d' % y
        x = X[ins_idx]
        for idx in range(len(x.indices)):
            line += ' %d:%.2f' % (x.indices[idx], x.data[idx])
        if ins_idx<95000:
            print >> train_file, line
        else:
            print >> test_file, line


