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

class UserAutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, EmbeddingSize)

        self.fc3 = nn.Linear(EmbeddingSize, 512)
        self.fc4 = nn.Linear(512, input_size)

    def forward(self, x):
        x_ = F.relu( self.fc1(x) )
        x_ = F.relu( self.fc2(x_) )
        x_ = F.relu( self.fc3(x_) )
        y = torch.sigmoid( self.fc4(x_) )
        return y

class DataLoader(easy_train.CommonDataLoader):
    def __init__(self, train):
        self.data = []
        max_movie_id = 0
        for uid, views in train:
            clicks = map(lambda x:int(x[0]), filter(lambda x:x[1]==1, views))
            if len(clicks)==0:
                continue
            
            max_movie_id = max(max_movie_id, max(clicks))
            self.data.append( clicks )

        self.user_count = len(self.data)
        self.movie_count = max_movie_id + 1
        pydev.log('user_count=%d' % self.user_count)
        pydev.log('max_movie_id=%d' % self.movie_count)

    def next_iter(self):
        s = random.sample(self.data, self.batch_size)
        x = torch.zeros( (self.batch_size, self.movie_count) )
        for idx, clicks in enumerate(s):
            for click_id in clicks:
                x[idx][click_id] = 1.

        return x
        

if __name__=='__main__':
    EmbeddingSize = 128
    train, valid, test = utils.readdata(sys.argv[1], test_num=1000)

    data = DataLoader(train)
    data.set_batch_size(100)

    model = UserAutoEncoder(data.movie_count, EmbeddingSize)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    def fwbp():
        x = data.next_iter()
        y_ = model.forward(x)
        loss = loss_fn(x, y_)
        loss.backward()
        return loss[0] / data.movie_count

    easy_train.easy_train(fwbp, data, optimizer, iteration_count=1000)

