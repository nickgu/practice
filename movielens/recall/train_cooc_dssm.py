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

class COOC_DSSM(nn.Module):
    def __init__(self, movie_count, embedding_size):
        # cosine(embedding, embedding)
        nn.Module.__init__(self)
        
        self.nid_emb = nn.Embedding(movie_count, embedding_size)

    def forward(self, a_nid, b_nid):
        # sum up embeddings.
        a_emb = self.nid_emb(a_nid)
        b_emb = self.nid_emb(b_nid)

        # activation: cosine.
        #y = F.cosine_similarity(a_emb, b_emb) * 0.5 + 0.5

        # activation: dot
        y = (a_emb * b_emb).sum(dim=1)
        y = F.sigmoid(y)
        return y


class DataGenerator:
    def __init__(self, train, device, epoch_count, batch_size):
        max_movie_id = 0

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
            self.data_count += len(views)

        self.train_iter_count = self.epoch_count * self.data_count / self.batch_size

        # self.train_data.write_over()
        self.movie_count = max_movie_id + 1
        pydev.log('max_movie_id=%d' % self.movie_count)
        pydev.log('data_count=%d' % self.data_count)

        
    def data_generator(self):
        epoch_num = 0
        while 1:
            # batch for embeddings.
            a_nids = []
            b_nids = []
            clicks = []

            epoch_num += 1
            pydev.info('Epoch %d' % epoch_num)
            for user, actions in self.train:
                actions = filter(lambda x:x[1] == 1, actions)
                
                for a_nid, _, _ in actions:
                    b_nid, _, _ = actions[random.randint(0, len(actions)-1)]
                    if a_nid == b_nid:
                        continue

                    a_nids.append(a_nid)
                    b_nids.append(b_nid)
                    clicks.append(1.)

                    a_nids.append(a_nid)
                    b_nids.append(random.randint(0, self.movie_count-1))
                    clicks.append(0.)

                    if len(clicks)>=self.batch_size:
                        yield (torch.tensor(a_nids).to(self.device),
                                torch.tensor(b_nids).to(self.device), 
                                torch.tensor(clicks).to(self.device))

                        a_nids = []
                        b_nids = []
                        clicks = []


if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    EmbeddingSize = 128
    EpochCount = 2
    BatchSize = 500

    pydev.info('EmbeddingSize=%d' % EmbeddingSize)
    pydev.info('Epoch=%d' % EpochCount)
    pydev.info('BatchSize=%d' % BatchSize)

    device = torch.device('cuda')

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    train, valid, test = utils.readdata(data_dir)
    data = DataGenerator(train, device, epoch_count=EpochCount, batch_size=BatchSize)

    model = COOC_DSSM(data.movie_count, EmbeddingSize).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    def fwbp():
        a_nids, b_nids, clicks = generator.next()

        clicks_ = model.forward(a_nids, b_nids)
        loss = loss_fn(clicks_, clicks)
        loss.backward()

        del a_nids, b_nids, clicks
        return loss[0]

    easy_train.easy_train(fwbp, optimizer, data.train_iter_count, loss_curve_output=file('log/loss.log', 'w'))

    nid_emb_fd = file('temp/cooc_dssm_out_emb.txt', 'w')
    easy_train.dump_embeddings(model.nid_emb, nid_emb_fd)


