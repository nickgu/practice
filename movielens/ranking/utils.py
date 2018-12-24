#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import tqdm

import sklearn
from sklearn import metrics

def readfile(fd, test_num=-1):
    data = []
    for line in fd.readlines():
        uid, iid, score = line.split(',')
        uid = int(uid)
        iid = int(iid)
        score = int(score)
        data.append( (uid, iid, score))
        if test_num>0 and len(data)>=test_num:
            break
    return data

def readdata(dir, test_num=-1):
    # return data:
    #   train/valid: [(uid, iid, score), ..]
    #   test: [(uid, iid), (uid, iid), ..]

    print >> sys.stderr, 'load [%s/]' % dir
    train = readfile(file(dir + '/train'), test_num)
    valid = readfile(file(dir + '/valid'))
    test = readfile(file(dir + '/test'))
    
    print >> sys.stderr, 'load over'
    return train, valid, test

def measure(predictor, test, debug=False):

    progress = tqdm.tqdm(test)
    y = []
    y_ = []
    for uid, iid, score in progress:
        pred_score = predictor(uid, iid)
        
        y.append( score )
        y_.append( pred_score )

    auc = metrics.roc_auc_score(y, y_)
    pydev.info('Test AUC: %.3f' % auc)
    

if __name__=='__main__':
    train, valid, test = readdata(sys.argv[1])
    print len(train)
    print len(valid)
    print len(test)
