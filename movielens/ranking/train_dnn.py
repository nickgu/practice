#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pydev
import utils

sys.path.append('../../learn_pytorch')
import easy_train

import tqdm
import numpy as np

import sklearn
from sklearn import metrics

class DNNRank(nn.Module):
    def __init__(self, user_count, item_count, embedding_size):
        # Ranking model:
        # input emb_size * 2 (embbag of input, emb of item to predict)
        # fc x 4
        nn.Module.__init__(self)
        
        self.uid_emb = nn.Embedding(user_count, embedding_size)
        self.iid_emb = nn.Embedding(item_count, embedding_size)

        self.fc1 = nn.Linear(embedding_size*2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, uid, iid):

        # sum up embeddings.
        user_emb = self.uid_emb(uid)
        item_emb = self.iid_emb(iid)

        x_ = torch.cat((user_emb, item_emb), 1)
        x_ = F.relu(self.fc1(x_))
        x_ = F.relu(self.fc2(x_))
        x_ = F.relu(self.fc3(x_))
        x_ = F.relu(self.fc4(x_))
        y = F.sigmoid( self.out(x_) )
        return y

class DataGenerator:
    def __init__(self, train, device, epoch_count, batch_size):
        max_movie_id = 0
        max_user_id = 0

        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.data_count = 0
        self.device = device
        self.train = train
        self.current_epoch = 0

        write_progress = tqdm.tqdm(train)
        for uid, iid, click in write_progress:
            max_movie_id = max(max_movie_id, iid)
            max_user_id = max(max_user_id, uid)
            self.data_count += 1

        self.train_iter_count = self.epoch_count * self.data_count / self.batch_size

        self.user_count = max_user_id + 1
        self.movie_count = max_movie_id + 1

        pydev.log('user_count=%d' % self.user_count)
        pydev.log('movie_count=%d' % self.movie_count)
        pydev.log('data_count=%d' % self.data_count)

        
    def data_generator(self):
        self.current_epoch= 0
        while 1:
            # batch for embeddings.
            user_ids = []
            item_ids = []
            clicks = []

            self.current_epoch += 1
            pydev.info('Epoch %d' % self.current_epoch)
            for uid, iid, click  in self.train:
                user_ids.append(uid)
                item_ids.append(iid)
                clicks.append(float(click))

                if len(clicks)>=self.batch_size:
                    yield (torch.tensor(user_ids).to(self.device),
                            torch.tensor(item_ids).to(self.device), 
                            torch.tensor(clicks).to(self.device))

                    user_ids = []
                    item_ids = []
                    clicks = []

if __name__=='__main__':
    autoarg = pydev.AutoArg()
    data_dir = autoarg.option('data', 'data/')
    model_save_path = autoarg.option('output', 'temp/dnn.pkl')

    TestNum = int(autoarg.option('testnum', -1))
    EmbeddingSize = int(autoarg.option('embed', 16))
    EpochCount = int(autoarg.option('epoch', 3))
    BatchSize = int(autoarg.option('batch', 1024))

    pydev.info('EmbeddingSize=%d' % EmbeddingSize)
    pydev.info('Epoch=%d' % EpochCount)
    pydev.info('BatchSize=%d' % BatchSize)

    device = torch.device('cuda')

    train, valid, test = utils.readdata(data_dir, test_num=TestNum)
    data = DataGenerator(train, device, epoch_count=EpochCount, batch_size=BatchSize)

    model = DNNRank(data.user_count, data.movie_count, EmbeddingSize).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    def test_validation():
        y = []
        y_ = []

        batch_size = 2048
        for begin in range(0, len(valid)-1, batch_size):
            output = model.forward(
                    torch.tensor(map(lambda x:x[0], valid[begin:begin+batch_size])).to(device),
                    torch.tensor(map(lambda x:x[1], valid[begin:begin+batch_size])).to(device),
                    )
            y += map(lambda x:x[2], valid[begin:begin+batch_size])
            y_ += output.view(-1).tolist()
        
        auc = metrics.roc_auc_score(y, y_)
        pydev.log('Valid AUC: %.3f' % auc)

    def fwbp():
        pre_epoch = data.current_epoch
        user_ids, item_ids, clicks = generator.next()
        if data.current_epoch > pre_epoch:
            # epoch complete.
            # test valid.
            test_validation()

        clicks_ = model.forward(user_ids, item_ids)
        loss = loss_fn(clicks_, clicks)
        loss.backward()

        del user_ids, item_ids, clicks
        return loss[0]

    pydev.info('Begin training..')
    easy_train.easy_train(fwbp, optimizer, data.train_iter_count, loss_curve_output=file('log/train_loss.log', 'w'))

    pydev.info('Saving model..')
    torch.save(model.state_dict(), model_save_path)


