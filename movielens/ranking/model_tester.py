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

import easy

class ModelTester(pydev.App):
    def __init__(self):
        pydev.App.__init__(self)
    
        self.debug=True
        self.device = torch.device('cuda')


    def load_uid_iid_data(self):
        pydev.info('Begin loading data..')
        # no need to load train.
        # only load 10000 train as test_of_train.
        self.test_of_train, self.valid, self.test = utils.readdata('data', test_num=10000)
        pydev.info('Load over')

    def test_uid_iid_model(self, model):
        self.load_uid_iid_data()
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

    def test_ins_data(self, model, slot_info):
        autoarg = pydev.AutoArg()
        input_filename = autoarg.option('test')
        batch_size = int(autoarg.option('batch', 20000))
        reader = easy.slot_file.SlotFileReader(input_filename)
        
        y = []
        y_ = []
        reading_count = 0
        while reader.epoch()<1:
            labels, slots = reader.next(batch_size)
            
            # make pytorch data.
            clicks = torch.Tensor(labels).to(self.device)
            dct = {}
            for item in slots:
                for slot, ids in item:
                    if slot not in dct:
                        # id_list, offset
                        dct[slot] = [[], []]

                    lst = dct[slot][0]
                    idx = dct[slot][1]
                    idx.append( len(lst) )
                    lst += ids

            x = []
            for slot, _ in slot_info:
                id_list, offset = dct.get(slot, [[], []])
                emb_pair = torch.tensor(id_list).to(self.device), torch.tensor(offset).to(self.device)
                x.append(emb_pair)

            clicks_ = model.forward(x)

            y += clicks.view(-1).tolist()
            y_ += clicks_.view(-1).tolist()

            pydev.log13('reading_count : %d' % reading_count)
            reading_count += 1

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

    def slot_dnn(self):
        import train_slot_dnn
        autoarg = pydev.AutoArg()

        EmbeddingSize = int(autoarg.option('emb', 32))
        slotinfo_filename = autoarg.option('s')
        model_path = autoarg.option('m')

        # temp get slot_info.
        slot_info = []
        for slot, slot_feanum in pydev.foreach_row(file(slotinfo_filename),format='si'):
            slot_info.append( (slot, slot_feanum) )

        model = train_slot_dnn.SlotDnnRank(slot_info, EmbeddingSize).to(self.device)
        model.load_state_dict( torch.load(model_path) )

        self.test_ins_data(model, slot_info)

if __name__=='__main__':
    app = ModelTester()
    app.run()

        
    
