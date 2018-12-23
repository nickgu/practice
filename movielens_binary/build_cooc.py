#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import random

import pydev
import utils

class CoocDict: 
    def __init__(self):
        self.cooc_dict = {}
        self.total_edge = 0

    def add(self, a, b):
        if a not in self.cooc_dict:
            self.cooc_dict[a] = {}
        self.cooc_dict[a][b] = self.cooc_dict[a].get(b, 0) + 1
        self.total_edge += 1

if __name__=='__main__':
    MinCooc = 0
    TestNum = -1
    if len(sys.argv)>1:
        TestNum = int(sys.argv[1])
    WindowSize = 5

    print >> sys.stderr, 'begin loading data..(testnum=%d)' % TestNum
    train, _, _ = utils.readdata('data', test_num=TestNum)
    print >> sys.stderr, 'load over'

    cooc_dict = CoocDict()
    for uid, items in train:
        items = filter(lambda x:x[1]==1, items)

        for idx in range(len(items)-WindowSize):
            a, _,_ = items[idx]
            for offset in range(WindowSize):
                b, _,_ = items[idx + 1 + offset]
                cooc_dict.add(a, b)
                cooc_dict.add(b, a)

    print >> sys.stderr, 'Total cooc: %d' % (cooc_dict.total_edge)

    fd = file('temp/cooc.txt', 'w')
    valid_cooc = 0
    heap = pydev.TopkHeap(100, lambda x:x[2])

    for key, items in cooc_dict.cooc_dict.iteritems():
        items = sorted(filter(lambda x:x[1]>MinCooc, items.iteritems()), key=lambda x:-x[1])
        if len(items) == 0:
            continue
        valid_cooc += sum(map(lambda x:x[1], items))
        print >> fd, '%s\t%s' % (key, ','.join(map(lambda x:'%s:%d'%(x[0],x[1]), items)) )

        for item, count in items:
            heap.push( (key, item, count) )

    for key, item, count in heap.sorted_data():
        print key, item, count

    print >> sys.stderr, 'Valid cooc: %d' % valid_cooc 
