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
        
        self.input_embbag = nn.EmbeddingBag(movie_count, embedding_size, mode='mean')
        self.nid_emb = nn.Embedding(movie_count, embedding_size)

        self.fc1 = nn.Linear(embedding_size*2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, input_nids, input_offset, click_item):

        # sum up embeddings.
        input_embs = self.input_embbag(input_nids, input_offset)
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

        self.batch_size = 200
        self.epoch = 0
        self.__offset = 0

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

        '''
        test_num = 10000
        self.x = self.x[:test_num]
        self.y = self.y[:test_num]

        print self.x
        print self.y
        '''

        self.movie_count = max_movie_id + 1
        pydev.log('max_movie_id=%d' % self.movie_count)
        pydev.log('data_count=%d' % len(self.x))
        
    def data_generator(self):
        while True:
            if self.__offset + self.batch_size > len(self.x):
                self.epoch += 1
                print >> sys.stderr, 'Epoch %d' % self.epoch

            input_nids = []
            input_offset = []
            y = []
            clicks = []
            
            for i in range(self.batch_size):
                idx = (self.__offset + i) % len(self.x)

                input_offset.append(len(input_nids))
                input_nids += self.x[idx]
                y.append( self.y[idx] )
                clicks.append( 1. )

                # negative sample.
                idx = random.randint(0, len(self.x)-1)
                input_offset.append(len(input_nids))
                input_nids += self.x[idx]
                y.append( random.randint(0, self.movie_count-1) )
                clicks.append( 0. )

            '''
            print input_nids
            print input_offset
            print y
            print clicks
            '''

            yield torch.tensor(input_nids), torch.tensor(input_offset), torch.tensor(y), torch.tensor(clicks)
            self.__offset = (self.__offset + self.batch_size) % len(self.x)



if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    EmbeddingSize = 128
    train, valid, test = utils.readdata(data_dir, test_num=10000)

    data = DataLoader(train)
    model = FC_DNN(data.movie_count, EmbeddingSize)
    #optimizer = optim.SGD(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    test_y = []
    test_y_ = []

    class Trainer: 
        def __init__(self):
            self.test_inputs = []

        def fwbp(self):
            input_nids, input_offset, y, clicks = generator.next()
            self.test_inputs.append( (input_nids, input_offset, y, clicks) )

            #print x, y, clicks
            clicks_ = model.forward(input_nids, input_offset, y)
        
            # temp test auc for testing.
            if sum(map(lambda x:len(x[3]), self.test_inputs))>=len(data.x)* 2:
                all_y_ = []
                all_y = []
                for a,b,c,y in self.test_inputs:
                    y_ = model.forward(a,b,c)
                    all_y_ += y_.squeeze().tolist()
                    all_y += y.tolist()

                easy_train.easy_auc(all_y_, all_y)
                self.test_inputs = []

            #print clicks_, clicks
            loss = loss_fn(clicks_, clicks)
            loss.backward()

            #print clicks, clicks_, loss[0]
            return loss[0]

    trainer = Trainer()

    easy_train.easy_train(trainer.fwbp, optimizer, 200000)

    torch.save(model.state_dict(), model_save_path)


