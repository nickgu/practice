#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils
import torch


class GroundTruthTest(pydev.App):
    def __init__(self):
        pydev.App.__init__(self)
    
        self.debug=True

        #TestNum = -1
        TestNum = -1

        pydev.info('Begin loading data..')
        self.train, self.valid, self.test = utils.readdata('data', test_num=TestNum)
        pydev.info('Load over')

    def nid_ctr(self):
        stat = {}
        global_disp = 0
        global_click = 0
        for uid, iid, click in self.train:
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
        utils.measure(predict_none_smooth, self.test, debug=self.debug)
        pydev.info('nid_ctr with static smooth')
        utils.measure(predict_static_smooth, self.test, debug=self.debug)
        pydev.info('nid_ctr with ratio smooth')
        utils.measure(predict_ratio_smooth, self.test, debug=self.debug)

    def lr(self):
        import train_lr
        model = train_lr.LRRank(138494, 131263, 8)
        model.load_state_dict( torch.load('temp/lr.pkl') )

        def predict(uid, iid, debug_fd):
            ret = model.forward(
                    torch.tensor([uid]), 
                    torch.tensor([iid]))

            print >> debug_fd, '(%s,%s): %s' % (uid, iid, ret[0])

            return ret[0].item()

        utils.measure(predict, self.test, self.debug)

    def dnn(self):
        import train_dnn
        model = train_dnn.DNNRank(138494, 131263, 16)
        model.load_state_dict( torch.load('temp/dnn.pkl') )

        def predict(uid, iid, debug_fd):
            ret = model.forward(
                    torch.tensor([uid]), 
                    torch.tensor([iid]))

            print >> debug_fd, '(%s,%s): %s' % (uid, iid, ret[0])

            return ret[0].item()

        utils.measure(predict, self.test, self.debug)


if __name__=='__main__':
    app = GroundTruthTest()
    app.run()

        
    
