#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils

def algor_test(train, valid, test, debug=False):
    def predict(uid, iid, debug_fd):
        return 0
    utils.measure(predict, test, debug=debug)

def algor_nid_ctr(train, valid, test, debug=False):
    stat = {}
    global_disp = 0
    global_click = 0
    for uid, iid, click in train:
        if iid not in stat:
            stat[iid] = [0, 0]

        stat[iid][0] += 1
        global_disp += 1
        if click:
            global_click += 1
            stat[iid][1] += 1

    global_click_ratio = global_click * 0.00001
    global_disp_ratio = global_disp * 0.00001

    def predict(uid, iid, debug_fd, smooth):
        s = stat.get(iid, [0, 0])
        if debug_fd:
            print >> debug_fd, 'stat\t%s\t%d\t%d' % (iid, s[0], s[1])
        if smooth==0:
            return s[1] * 1. / (s[0] + 0.1)
        elif smooth==1:
            return (s[1] + 1.) / (s[0] + 10.)
        elif smooth==2:
            return (s[1] + global_click_ratio) / (s[0] + global_disp_ratio)

    predict_none_smooth = lambda u,i,d:predict(u,i,d,smooth=0)
    predict_static_smooth = lambda u,i,d:predict(u,i,d,smooth=1)
    predict_ratio_smooth = lambda u,i,d:predict(u,i,d,smooth=2)

    pydev.info('nid_ctr with none smooth')
    utils.measure(predict_none_smooth, test, debug=debug)
    pydev.info('nid_ctr with static smooth')
    utils.measure(predict_static_smooth, test, debug=debug)
    pydev.info('nid_ctr with ratio smooth')
    utils.measure(predict_ratio_smooth, test, debug=debug)

if __name__=='__main__':
    TestNum = -1
    #TestNum = 100

    pydev.info('Begin loading data..')
    train, valid, test = utils.readdata('data', test_num=TestNum)
    pydev.info('Load over')

    pydev.info('Algor: NidCTR')
    algor_nid_ctr(train, valid, test, debug=True)



        
    
