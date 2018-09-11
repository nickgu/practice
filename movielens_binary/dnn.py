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

import numpy as np

class FC_DNN(nn.Module):
    def __init__(self, movie_count, embedding_size):
        nn.Module.__init__(self)
        
        self.input_emb = nn.Embedding(movie_count, embedding_size)
        self.nid_emb = nn.Embedding(movie_count, embedding_size)

        self.fc1 = nn.Linear(embedding_size*2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, input_nids, click_item):

        # sum up embeddings.
        input_embs = self.input_emb(input_nids).sum(1)
        y_emb = self.nid_emb(click_item)

        x_ = torch.cat((input_embs, y_emb), 1)
        x_ = F.relu(self.fc1(x_))
        x_ = F.relu(self.fc2(x_))
        x_ = F.relu(self.fc3(x_))
        y = F.sigmoid( self.fc4(x_) )
        return y


class DataLoader:
    def __init__(self, train):
        self.x = []
        self.y = []
        max_movie_id = 0

        for uid, views in train:
            clicks = map(lambda x:int(x[0]), filter(lambda x:x[1]==1, views))
            if len(clicks)==0:
                continue
            
            max_movie_id = max(max_movie_id, max(clicks))

            for idx, click in enumerate(clicks):
                x = clicks[:idx]
                y = clicks[idx]
                if len(x)<3:
                    continue
                self.x.append(x)
                self.y.append(y)

        self.movie_count = max_movie_id + 1
        pydev.log('max_movie_id=%d' % self.movie_count)
        pydev.log('data_count=%d' % len(self.x))
        
    def data_generator(self):
        epoch = 0
        while 1:
            epoch += 1
            print >> sys.stderr, 'Epoch %d' % epoch
            for idx in range(len(self.x)):
                x = []
                y = []
                clicks = []
                sample_size = 3
                for i in range(sample_size):
                    x.append( self.x[idx] )
                    if i == 0:
                        y.append( self.y[idx] )
                        clicks.append( 1. )
                    else:
                        y.append( random.choice(range(self.movie_count)) )
                        clicks.append( 0. )

                yield torch.tensor(x), torch.tensor(y), torch.tensor(clicks)

if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    EmbeddingSize = 128
    train, valid, test = utils.readdata(data_dir, test_num=1000)

    data = DataLoader(train)
    model = FC_DNN(data.movie_count, EmbeddingSize)
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    test_y = []
    test_y_ = []

    class Trainer: 
        def __init__(self):
            self.test_y = []
            self.test_y_ = []

        def fwbp(self):
            x, y, clicks = generator.next()

            #print x, y, clicks
            clicks_ = model.forward(x, y)
        
            # temp test auc for testing.
            for idx in range(len(clicks)):
                self.test_y.append( clicks[idx].long().item() )
                self.test_y_.append( clicks_[idx].item() )
            if len(self.test_y)>=1000:
                easy_train.easy_auc(self.test_y_, self.test_y)
                self.test_y = []
                self.test_y_ = []

            #print clicks_, clicks
            loss = loss_fn(clicks_, clicks)
            loss.backward()

            #print clicks, clicks_, loss[0]
            return loss[0]

    trainer = Trainer()

    easy_train.easy_train(trainer.fwbp, optimizer, 200000)

    torch.save(model.state_dict(), model_save_path)


