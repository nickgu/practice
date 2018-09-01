#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import utils

if __name__=='__main__':
    print >> sys.stderr, 'begin loading data..'
    train, valid, test = utils.readdata('data')
    print >> sys.stderr, 'load over'

    stat = {}
    for uid, items in train:
        for iid, score, ts in items:
            if iid not in stat:
                stat[iid] = [0, 0]

            stat[iid][score] += 1

    top = sorted(stat.iteritems(), key=lambda x:-x[1][1])[:10]
    print 'stat over'
    print top

    def predict(uid, items):
        return map(lambda x:x[0], top)

    utils.measure(predict, test)


        
    
