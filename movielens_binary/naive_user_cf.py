#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils

def algor_naive_usercf(train, valid, test, topN):
    index = {}
    readlist = {}
    for uid, items in train:
        rl = map(lambda x:x[0], filter(lambda x:x[1]==1, items))

        readlist[uid] = rl
        for iid in rl:
            if iid not in index:
                index[iid] = []
            index[iid].append(uid)

    print >> sys.stderr, 'index build ok'
    
    def predict(uid, items):
        readset = set(map(lambda x:x[0], items))

        sim_users = {}
        for iid, score, ts in items:
            rlist = index.get(iid, [])
            for user in rlist:
                sim_users[user] = sim_users.get(user, 0) + 1

        sim_users_list = sorted(sim_users.iteritems(), key=lambda x:-x[1])[:20]
        #print sim_users_list

        s_set = {}
        for sim_user, sim_count in sim_users_list:
            for item in readlist[sim_user]:
                if item in readset:
                    continue
                s_set[item] = s_set.get(item, 0)+1
        ret = map(lambda x:x[0], sorted(s_set.iteritems(), key=lambda x:-x[1])[:topN])
        #print ret
        return ret

    utils.measure(predict, test, debug=False)


if __name__=='__main__':
    TopN = 10
    TestNum = -1

    print >> sys.stderr, 'begin loading data..'
    train, valid, test = utils.readdata('data', test_num=TestNum)
    print >> sys.stderr, 'load over'
    
    print >> sys.stderr, 'Algor: Naive_UserCF'
    algor_naive_usercf(train, valid, test, TopN)

