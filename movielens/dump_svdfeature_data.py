#! /bin/env python
# encoding=utf-8
# author: nickgu
#
#   Usage:
#       python dump_svdfeature_data.py <movielens_path>
#       
#   example:
#      movielens_user_data format:
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
    U = []
    I = []
    Y = []
    for userid, itemid, rating in data:
        d = { "user_id": str(userid) } 
        d = info.process_user(userid, itemid, d)
        U.append(d)

        d = { "movie_id": str(itemid) }
        d = info.process_movie(userid, itemid, d)
        I.append(d)

        Y.append( rating )

    v = DictVectorizer()
    U = v.fit_transform(U)
    v = DictVectorizer()
    I = v.fit_transform(I)

    print >> sys.stderr, 'load data over U=%d I=%d test=%d' % (U.shape[0], I.shape[0], len(Y))
    return U, I, Y

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: dump_svdfeature_data.py <ml_data_path>'
        sys.exit(-1)

    U, I, Y = load_data(path=sys.argv[1] + '/')

    max_user_feature = 0
    max_movie_feature = 0

    train_file = file('data/svdf_train.txt', 'w')
    test_file = file('data/svdf_test.txt', 'w')
    for ins_idx, y in enumerate(Y):
        # line format:
        #   label g_num u_num i_num <global_features> <user_features> <item_features>
        line = '%d' % y
        f_u = U[ins_idx]
        f_i = I[ins_idx]
        line += ' %d %d %d' % (0, len(f_u.indices), len(f_i.indices))
        
        # global feature.
        pass

        # user feature.
        for idx in range(len(f_u.indices)):
            line += ' %d:%.2f' % (f_u.indices[idx], f_u.data[idx])
            if max_user_feature < f_u.indices[idx]:
                max_user_feature = f_u.indices[idx]

        # item feature.
        for idx in range(len(f_i.indices)):
            line += ' %d:%.2f' % (f_i.indices[idx], f_i.data[idx])
            if max_movie_feature < f_i.indices[idx]:
                max_movie_feature = f_i.indices[idx]

        # output file.
        if ins_idx<95000:
            print >> train_file, line
        else:
            print >> test_file, line

    print >> sys.stderr, 'user_feature_count = %d' % (max_user_feature + 1)
    print >> sys.stderr, 'movie_feature_count = %d' % (max_movie_feature + 1)


