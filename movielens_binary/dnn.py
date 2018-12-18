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

class TrainData:
    def __init__(self, filename, mode='w'):
        self.__fd = file(filename, mode)
        self.__filename = filename

    def write(self, x, y):
        print >> self.__fd, '%s\t%s' % (','.join(map(lambda x:str(x), x)), y)

    def write_over(self):
        self.__fd.close()
        self.__fd = file(self.__filename, 'r')

    def read(self):
        line = self.__fd.readline()
        if line == '':
            print >> sys.stderr, 'Complete epoch.'
            self.__fd = file(self.__filename, 'r')
            line = self.__fd.readline()
            if line == '':
                raise Exception('Cannot reopen file to read.')
        x, y = line.strip().split('\t')
        x = map(lambda x:int(x), x.split(','))
        y = int(y)
        return x, y

    def read_batch(self, batch_size=100):
        rx = []
        ry = []
        for i in range(batch_size):
            x, y = self.read()
            rx.append(x)
            ry.append(y)
        return rx, ry
        

class FC_DNN(nn.Module):
    def __init__(self, movie_count, embedding_size):
        # Ranking model:
        # input emb_size * 2 (embbag of input, emb of item to predict)
        # fc x 4
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
    def __init__(self, train, device):
        max_movie_id = 0

        self.batch_size = 200
        self.data_count = 0
        self.device = device

        filename = 'temp/train.data'

        self.train_data = TrainData(filename, 'r')
        # self.train_data = TrainData(filename)

        write_progress = tqdm.tqdm(train)
        for uid, views in write_progress:
            clicks = map(lambda x:int(x[0]), filter(lambda x:x[1]==1, views))
            if len(clicks)==0:
                continue
            
            max_movie_id = max(max_movie_id, max(clicks))

            for idx, click in enumerate(clicks):
                x = clicks[:idx]
                y = clicks[idx]
                if len(x)<3:
                    continue
                
                #self.train_data.write(x, y)
                self.data_count += 1

        # self.train_data.write_over()
        self.movie_count = max_movie_id + 1
        pydev.log('max_movie_id=%d' % self.movie_count)
        pydev.log('data_count=%d' % self.data_count)

        
    def data_generator(self):
        while True:
            # batch for embeddings.
            input_nids = []
            input_offset = []
            y = []
            clicks = []
            
            for i in range(self.batch_size):
                a, b = self.train_data.read()

                input_offset.append(len(input_nids))
                input_nids += a
                y.append( b )
                clicks.append( 1. )

                # negative sample.
                input_offset.append(len(input_nids))
                input_nids += a
                y.append( random.randint(0, self.movie_count-1) )
                clicks.append( 0. )

            '''
            print input_nids
            print input_offset
            print y
            print clicks
            '''

            yield torch.tensor(input_nids).to(self.device),torch.tensor(input_offset).to(self.device), torch.tensor(y).to(self.device), torch.tensor(clicks).to(self.device)
            del input_nids
            del input_offset
            del y
            del clicks



if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    device = torch.device('cuda')

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    EmbeddingSize = 32

    train, valid, test = utils.readdata(data_dir)

    data = DataLoader(train, device)
    del train

    model = FC_DNN(data.movie_count, EmbeddingSize).to(device)
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
            #self.test_inputs.append( (input_nids, input_offset, y, clicks) )

            #print x, y, clicks
            clicks_ = model.forward(input_nids, input_offset, y)
        
            # temp test auc for testing.
            '''
            if sum(map(lambda x:len(x[3]), self.test_inputs))>=data.data_count * 2:
                all_y_ = []
                all_y = []
                for a,b,c,y in self.test_inputs:
                    y_ = model.forward(a,b,c)
                    all_y_ += y_.squeeze().tolist()
                    all_y += y.tolist()

                easy_train.easy_auc(all_y_, all_y)
                self.test_inputs = []
            '''

            #print clicks_, clicks
            loss = loss_fn(clicks_, clicks)
            loss.backward()

            #print clicks, clicks_, loss[0]
            del input_nids, input_offset, y, clicks
            return loss[0]

    trainer = Trainer()

    easy_train.easy_train(trainer.fwbp, optimizer, 200000)

    torch.save(model.state_dict(), model_save_path)

