#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils
import torch
import tqdm

import sklearn
from sklearn import metrics

class ModelTester(pydev.App):
    def __init__(self):
        pydev.App.__init__(self)
    
        self.debug=True
        self.device = torch.device('cuda')

        pydev.info('Begin loading data..')
        # no need to load train.
        # only load 10000 train as test_of_train.
        self.test_of_train, self.valid, self.test = utils.readdata('data', test_num=10000)
        pydev.info('Load over')

    def test_uid_iid_model(self, model):
        y = []
        y_ = []

        batch_size = 2048
        for begin in tqdm.tqdm(range(0, len(self.test)-1, batch_size)):
            output = model.forward(
                    torch.tensor(map(lambda x:x[0], self.test[begin:begin+batch_size])).to(self.device),
                    torch.tensor(map(lambda x:x[1], self.test[begin:begin+batch_size])).to(self.device),
                    )
            y += map(lambda x:x[2], self.test[begin:begin+batch_size])
            y_ += output.view(-1).tolist()
        
        auc = metrics.roc_auc_score(y, y_)
        print 
        pydev.log('Valid AUC: %.3f' % auc)

    def lr(self):
        import train_lr
        model = train_lr.LRRank(138494, 131263, 8)

        auto_arg = pydev.AutoArg()
        model_path = auto_arg.option('model', 'temp/lr.pkl')

        model.load_state_dict( torch.load(model_path) )
        model.to(self.device)
        self.test_uid_iid_model(model)


    def dnn(self):
        import train_dnn
        model = train_dnn.DNNRank(138494, 131263, 16)

        auto_arg = pydev.AutoArg()
        model_path = auto_arg.option('model', 'temp/dnn.pkl')

        model.load_state_dict( torch.load(model_path) )
        model.to(self.device)
        self.test_uid_iid_model(model)

if __name__=='__main__':
    app = ModelTester()
    app.run()

        
    
