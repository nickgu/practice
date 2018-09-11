#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import random
import torch

import pydev
import utils
import dnn

def algor_dnn_predict(train, valid, test, topN):
    movie_id_set = set()
    for uid, items in train:
        for iid, score, ts in items:
            if iid not in movie_id_set:
                movie_id_set.add(iid)

    model = dnn.FC_DNN(128716, 128)
    model.load_state_dict(torch.load('temp/dnn.pkl'))

    movie_ids = list(movie_id_set)


    def predict(uid, items):
        readset = set(map(lambda x:x[0], items))

        ans = []
        for item in map(lambda x:x[0], top):
            if item in readset:
                continue
            ans.append(item)
            if len(ans) == topN:
                break
        return ans[:topN]

    utils.measure(predict, test)


if __name__=='__main__':
    TopN = 10
    TestNum = 100

    print >> sys.stderr, 'begin loading data..'
    train, valid, test = utils.readdata('data', test_num=TestNum)
    print >> sys.stderr, 'load over'


    model = dnn.FC_DNN(128716, 128)
    model.load_state_dict(torch.load('temp/dnn.pkl'))

    for uid, items in test:
        items = map(lambda x:int(x[0]), filter(lambda x:x[1]==1, items))
    
        m = len(items) / 2
        inp = items[:m]
        ans = items[m:] 
        if len(inp)==0 or len(ans)==0:
            continue

        x = []
        y = []
        for item in ans:
            x.append( inp )
            y.append( item )
        for i in range(20):
            x.append( inp )
            y.append( random.randint(0, 128716) )
        x = torch.tensor(x)
        y = torch.tensor(y)
        click = model.forward(x, y)
        print x, y, click
        sys.stdin.read()

    #print >> sys.stderr, 'Algor: DNN_Predict'
    #algor_dnn_predict(train, valid, test, TopN)

