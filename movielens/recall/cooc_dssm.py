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

class COOC_DSSM(nn.Module):
    def __init__(self, movie_count, embedding_size):
        nn.Module.__init__(self)
        
        #self.input_emb = nn.Embedding(movie_count, embedding_size)
        self.nid_emb = nn.Embedding(movie_count, embedding_size)

    def forward(self, input_item, cooc_item):
        x_emb = self.nid_emb(input_item)
        y_emb = self.nid_emb(cooc_item)

        y = F.cosine_similarity(x_emb, y_emb) * 0.5 + 0.5
        return y

class TrainData:
    def __init__(self, filename):
        self.__fd = file(filename, 'w')
        self.__filename = filename

    def write(self, x, y):
        print >> self.__fd, '%s\t%s' % (x, y)

    def write_over(self):
        self.__fd.close()
        self.__fd = file(self.__filename, 'r')

    def read(self):
        line = self.__fd.readline()
        if line == '':
            print >> sys.stderr, 'Complete epoch.'
            self.__fd = file(self.__filename, 'r')
        x, y = line.strip().split('\t')
        x = int(x)
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
        

class DataLoader:
    def __init__(self, train, device):
        max_movie_id = 0

        self.device = device
        self.batch_size = 200
        self.data_count = 0
        self.window_size = 2

        filename = 'temp/train.cooc_dnn.data'
        self.train_data = TrainData(filename)

        write_progress = tqdm.tqdm(train)
        for uid, views in write_progress:
            clicks = map(lambda x:int(x[0]), filter(lambda x:x[1]==1, views))
            if len(clicks)==0:
                continue
            
            max_movie_id = max(max_movie_id, max(clicks))

            for idx, x in enumerate(clicks):
                for offset in range(self.window_size):
                    if idx+offset+1 >= len(clicks):
                        continue
                    y = clicks[idx + offset + 1]

                    self.train_data.write(x, y)
                    self.data_count += 1

        self.train_data.write_over()
        self.movie_count = max_movie_id + 1
        pydev.log('max_movie_id=%d' % self.movie_count)
        pydev.log('data_count=%d' % self.data_count)

        
    def data_generator(self):
        while True:
            x = []
            y = []
            clicks = []
            
            for i in range(self.batch_size):
                a, b = self.train_data.read()

                x.append( a )
                y.append( b )
                clicks.append( 1. )

                # negative sample.
                x.append( a )
                y.append( random.randint(0, self.movie_count-1) )
                clicks.append( 0. )

            yield torch.tensor(x).to(self.device), torch.tensor(y).to(self.device), torch.tensor(clicks).to(self.device)

if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ncooc_dssm.py <datadir> <model>'
        sys.exit(-1)

    device = torch.device('cuda')

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    EpochCount = 1
    EmbeddingSize = 32
    test_num = -1

    train, valid, test = utils.readdata(data_dir, test_num=test_num)

    data = DataLoader(train, device)
    del train

    model = COOC_DSSM(data.movie_count, EmbeddingSize).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    test_y = []
    test_y_ = []

    class Trainer: 
        def __init__(self):
            self.test_inputs = []

        def fwbp(self):
            x, y, clicks = generator.next()
            #self.test_inputs.append( (x, y, clicks) )

            #print x, y, clicks
            clicks_ = model.forward(x, y)
        
            # temp test auc for testing.
            '''
            if sum(map(lambda x:len(x[0]), self.test_inputs))>= 100000:
                all_y_ = []
                all_y = []
                for a,b,y in self.test_inputs:
                    y_ = model.forward(a,b)
                    all_y_ += y_.squeeze().tolist()
                    all_y += y.tolist()

                easy_train.easy_auc(all_y_, all_y)
                self.test_inputs = []
            '''

            #print clicks_, clicks
            loss = loss_fn(clicks_, clicks)
            loss.backward()

            #print clicks, clicks_, loss[0]
            return loss[0]

    trainer = Trainer()

    iter_count = EpochCount * data.data_count / data.batch_size
    easy_train.easy_train(trainer.fwbp, optimizer, iter_count)

    torch.save(model.state_dict(), model_save_path)

    input_emb_fd = file('temp/input_emb.txt', 'w')
    easy_train.dump_embeddings(model.input_emb, input_emb_fd)

    nid_emb_fd = file('temp/nid_emb.txt', 'w')
    easy_train.dump_embeddings(model.nid_emb, nid_emb_fd)


