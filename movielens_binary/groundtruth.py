#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils

def algor_hot(train, valid, test, topN):
    stat = {}
    for uid, items in train:
        for iid, score, ts in items:
            if iid not in stat:
                stat[iid] = [0, 0]

            stat[iid][score] += 1

    top = sorted(stat.iteritems(), key=lambda x:-x[1][1])[:topN]
    print 'stat over'
    print top

    def predict(uid, items):
        return map(lambda x:x[0], top)

    utils.measure(predict, test)

def algor_cooc(train, valid, test, topN):
    # using dict built by build_cooc.py
    fd = file('temp/cooc.txt') 

    cooc_dict = {}
    for key, items in pydev.foreach_row(fd):
        items = map(lambda x:(x[0], int(x[1])), map(lambda x:x.split(':'), items.split(',')))
        cooc_dict[key] = items
    print >> sys.stderr, 'cooc load over'

    def predict(uid, items):
        local_stat = {}
        for item, score, _ in items:
            cooc_items = cooc_dict.get(item, [])
            for c_item, c_count in cooc_items:
                local_stat[c_item] = local_stat.get(c_item, 0) + c_count

        ans = map(lambda x:x[0], sorted(local_stat.iteritems(), key=lambda x:-x[1])[:topN])
        return ans

    utils.measure(predict, test, debug=False)
        

if __name__=='__main__':
    TopN = 10
    TestNum = -1

    print >> sys.stderr, 'begin loading data..'
    train, valid, test = utils.readdata('data', test_num=TestNum)
    print >> sys.stderr, 'load over'

    algor_hot(train, valid, test, TopN)
    algor_cooc(train, valid, test, TopN)


        
    
