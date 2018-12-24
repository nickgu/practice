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

sys.path.append('../learn_pytorch')
import easy_train

import tqdm
import numpy as np

class UID_NID_DSSM(nn.Module):
    def __init__(self, user_count, movie_count, embedding_size):
        # cosine(user_embedding, item_embedding)
        nn.Module.__init__(self)
        
        self.user_emb = nn.Embedding(user_count, embedding_size, max_norm=EmbeddingSize*0.1)
        self.item_emb = nn.Embedding(movie_count, embedding_size, max_norm=EmbeddingSize*0.1)

    def forward(self, uid, nid):
        # sum up embeddings.
        u_emb = self.user_emb(uid)
        i_emb = self.item_emb(nid)

        # activation: dot
        y = (u_emb * i_emb).sum(dim=1)
        y = F.sigmoid(y)
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
        for uid, views in write_progress:
            clicks = map(lambda x:int(x[0]), filter(lambda x:x[1]==1, views))
            if len(clicks)==0:
                continue
            
            max_movie_id = max(max_movie_id, max(clicks))
            max_user_id = max(max_user_id, uid)
            self.data_count += len(views)

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
            for user_id, actions in self.train:
                actions = filter(lambda x:x[1] == 1, actions)
                
                for item_id, _, _ in actions:

                    user_ids.append(user_id)
                    item_ids.append(item_id)
                    clicks.append(1.)

                    user_ids.append(user_id)
                    item_ids.append(random.randint(0, self.movie_count-1))
                    clicks.append(0.)

                    if len(clicks)>=self.batch_size:
                        yield (torch.tensor(user_ids).to(self.device),
                                torch.tensor(item_ids).to(self.device), 
                                torch.tensor(clicks).to(self.device))

                        user_ids = []
                        item_ids = []
                        clicks = []


if __name__=='__main__':
    if len(sys.argv)!=2:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir>'
        sys.exit(-1)

    TestNum = 1000
    EmbeddingSize = 256
    EpochCount = 120
    BatchSize = 256

    pydev.info('EmbeddingSize=%d' % EmbeddingSize)
    pydev.info('Epoch=%d' % EpochCount)
    pydev.info('BatchSize=%d' % BatchSize)

    device = torch.device('cuda')

    data_dir = sys.argv[1]

    train, valid, test = utils.readdata(data_dir, test_num=TestNum)
    data = DataGenerator(train, device, epoch_count=EpochCount, batch_size=BatchSize)

    model = UID_NID_DSSM(data.user_count, data.movie_count, EmbeddingSize).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    def fwbp():
        user_ids, item_ids, clicks = generator.next()

        clicks_ = model.forward(user_ids, item_ids)
        loss = loss_fn(clicks_, clicks)
        loss.backward()

        del user_ids, item_ids, clicks
        return loss[0]

    easy_train.easy_train(fwbp, optimizer, data.train_iter_count, loss_curve_output=file('log/loss.log', 'w'))

    nid_emb_fd = file('temp/user_item_dssm.txt', 'w')
    easy_train.dump_embeddings(model.item_emb, nid_emb_fd)


