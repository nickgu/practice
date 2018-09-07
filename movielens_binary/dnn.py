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

class FCDNN(nn.Module):
    def __init__(self, movie_count, embedding_size):
        nn.Module.__init__(self)
        
        self.input_emb = nn.Embedding(movie_count, embedding_size)
        self.nid_emb = nn.Embedding(movie_count, embedding_size)

        self.fc1 = nn.Linear(embedding_size*2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, input_nids, click_items):

        # sum up embeddings.
        input_embs = self.input_emb(input_nids).sum(1)
        y_emb = self.nid_emb(click_items)

        x_ = torch.cat((input_embs, y_emb), 1)
        x_ = F.relu(self.fc1(x_))
        x_ = F.relu(self.fc2(x_))
        x_ = F.relu(self.fc3(x_))
        y = F.relu(self.fc4(x_))
        return y


class DataLoader(easy_train.CommonDataLoader):
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
        
    def next_iter(self):
        idx = random.choice(range(len(self.x)))
        x = []
        y = []
        clicks = []
        for i in range(10):
            x.append( self.x[idx] )
            if i == 0:
                y.append( self.y[idx] )
                clicks.append( 1 )
            else:
                y.append( random.choice(range(self.movie_count)) )
                clicks.append( 0 )

        return torch.tensor(x), torch.tensor(y), torch.tensor(clicks)

if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    EmbeddingSize = 128
    train, valid, test = utils.readdata(data_dir, test_num=1000)

    data = DataLoader(train)
    data.set_batch_size(100)

    model = FCDNN(data.movie_count, EmbeddingSize)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    def fwbp():
        x, y, clicks = data.next_iter()
        #print x, y, clicks
        clicks_ = model.forward(x, y)
        #print clicks, clicks_
        loss = loss_fn(clicks_, clicks)
        loss.backward()
        return loss[0] / data.movie_count

    easy_train.easy_train(fwbp, data, optimizer, iteration_count=1000)

    torch.save(model.state_dict(), model_save_path)


