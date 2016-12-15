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


def load_data(filename, feature_manager):
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

    G = []
    U = []
    I = []
    Y = []
    for userid, itemid, rating in data:
        d = {}
        d = feature_manager.process_global(userid, itemid, d)
        G.append(d)

        d = { "user_id": str(userid) } 
        d = feature_manager.process_user(userid, itemid, d)
        U.append(d)

        d = { "movie_id": str(itemid) }
        d = feature_manager.process_movie(userid, itemid, d)
        I.append(d)

        Y.append( rating )

    return G, U, I, Y


def dump_data(G, U, I, Y, output_filename):
    max_global_feature = 0
    max_user_feature = 0
    max_movie_feature = 0

    output_file = file(output_filename, 'w')
    for ins_idx, y in enumerate(Y):
        # line format:
        #   label g_num u_num i_num <global_features> <user_features> <item_features>
        line = '%d' % y
        f_g = G[ins_idx]
        f_u = U[ins_idx]
        f_i = I[ins_idx]
        line += ' %d %d %d' % (len(f_g.indices), len(f_u.indices), len(f_i.indices))
        
        # global feature.
        for idx in range(len(f_g.indices)):
            line += ' %d:%.2f' % (f_g.indices[idx], f_g.data[idx])
            if max_global_feature < f_g.indices[idx]:
                max_global_feature = f_g.indices[idx]

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

        print >> output_file, line

    print max_global_feature, max_user_feature, max_movie_feature


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: dump_svdfeature_data.py <ml_data_path>'
        sys.exit(-1)

    path = sys.argv[1] + '/'
    feature_manager = Info(path)

    train_G, train_U, train_I, train_Y = load_data(filename = path + '/ua.base', feature_manager=feature_manager)
    test_G, test_U, test_I, test_Y = load_data(filename = path + '/ua.test', feature_manager=feature_manager)

    global_dv = DictVectorizer()
    user_dv = DictVectorizer()
    item_dv = DictVectorizer()
    global_dv.fit(train_G + test_G)
    user_dv.fit(train_U + test_U)
    item_dv.fit(train_I + test_I)

    train_G = global_dv.transform( train_G )
    test_G = global_dv.transform( test_G )

    train_U = user_dv.transform( train_U )
    test_U = user_dv.transform( test_U )

    train_I = item_dv.transform( train_I )
    test_I = item_dv.transform( test_I )

    dump_data(train_G, train_U, train_I, train_Y, 'data/svdf_train.txt')
    dump_data(test_G, test_U, test_I, test_Y, 'data/svdf_test.txt')

    #print >> sys.stderr, 'user_feature_count = %d' % (max_user_feature + 1)
    #print >> sys.stderr, 'movie_feature_count = %d' % (max_movie_feature + 1)


