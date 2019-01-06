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

class LRRank(nn.Module):
    def __init__(self, user_count, item_count, embedding_size):
        # Ranking model:
        # input emb_size * 2 (embbag of input, emb of item to predict)
        # fc x 4
        nn.Module.__init__(self)
        
        pydev.info('user_count=%d' % user_count)
        pydev.info('item_count=%d' % item_count)
        pydev.info('embedding=%d' % embedding_size)

        self.uid_emb = nn.Embedding(user_count, embedding_size)
        self.iid_emb = nn.Embedding(item_count, embedding_size)
        self.lr = nn.Linear(embedding_size*2, 1)

    def forward(self, uid, iid):
        # sum up embeddings.
        user_emb = self.uid_emb(uid)
        item_emb = self.iid_emb(iid)
        x_ = torch.cat((user_emb, item_emb), 1)
        y = F.sigmoid( self.lr(x_) )
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
        epoch_num = 0
        while 1:
            # batch for embeddings.
            user_ids = []
            item_ids = []
            clicks = []

            epoch_num += 1
            pydev.info('Epoch %d' % epoch_num)
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
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    TestNum = 10000
    EmbeddingSize = 8
    EpochCount = 4
    BatchSize = 10000
    device = torch.device('cpu')

    pydev.info('EmbeddingSize=%d' % EmbeddingSize)
    pydev.info('Epoch=%d' % EpochCount)
    pydev.info('BatchSize=%d' % BatchSize)


    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    train, valid, test = utils.readdata(data_dir, test_num=TestNum)
    data = DataGenerator(train, device, epoch_count=EpochCount, batch_size=BatchSize)

    model = LRRank(data.user_count, data.movie_count, EmbeddingSize).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    test_y = []
    test_y_ = []

    def fwbp():
        user_ids, item_ids, clicks = generator.next()

        clicks_ = model.forward(user_ids, item_ids)
        #print 'size'
        #print clicks.size()
        #print clicks_.size()
        loss = loss_fn(clicks_, clicks)
        loss.backward()

        del user_ids, item_ids, clicks
        return loss[0]

    pydev.info('Begin training..')
    easy_train.easy_train(fwbp, optimizer, data.train_iter_count, loss_curve_output=file('log/train_loss.log', 'w'))

    pydev.info('Saving model..')
    torch.save(model.state_dict(), model_save_path)


