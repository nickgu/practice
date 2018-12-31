#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils
import torch

class ModelTester(pydev.App):
    def __init__(self):
        pydev.App.__init__(self)
    
        self.debug=True

        pydev.info('Begin loading data..')
        # no need to load train.
        # only load 10000 train as test_of_train.
        self.test_of_train, self.valid, self.test = utils.readdata('data', test_num=10000)
        pydev.info('Load over')

    def lr(self):
        import train_lr
        model = train_lr.LRRank(138494, 131263, 8)

        auto_arg = pydev.AutoArg()
        model_path = auto_arg.option('model', 'temp/lr.pkl')

        model.load_state_dict( torch.load(model_path) )


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
    app = ModelTester()
    app.run()

        
    
