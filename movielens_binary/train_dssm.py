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

class DSSM(nn.Module):
    def __init__(self, movie_count, embedding_size):
        # Ranking model:
        # input emb_size * 2 (embbag of input, emb of item to predict)
        # fc x 4
        nn.Module.__init__(self)
        
        self.input_embbag = nn.EmbeddingBag(movie_count, embedding_size, mode='mean')
        self.nid_emb = nn.Embedding(movie_count, embedding_size)

    def forward(self, input_nids, input_offset, click_item):

        # sum up embeddings.
        inputs_emb = self.input_embbag(input_nids, input_offset)
        y_emb = self.nid_emb(click_item)

        #y = F.cosine_similarity(inputs_emb, y_emb) * 0.5 + 0.5
        y = F.cosine_similarity(inputs_emb, y_emb) * 0.5 + 0.5
        return y


class DataLoader:
    def __init__(self, train, device):
        max_movie_id = 0

        self.epoch_count = 10
        self.batch_size = 500
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
        InputSize = 10
        for epoch_num in range(self.epoch_count):
            # batch for embeddings.
            input_nids = []
            input_offset = []
            y = []
            clicks = []

            pydev.info('Epoch %d' % epoch_num)
            for user, actions in self.train:
                # put all to input.
                input = map(lambda x:x[0], actions)
                #input = map(lambda x:x[0], filter(lambda x:x[1]==1, actions))
                for item, click, _ in actions:
                    input_offset.append(len(input_nids))
                    input_nids += input
                    y.append( item )
                    clicks.append( float(click) )

                    #print input_nids, input_offset, y, clicks
                    if len(clicks)>=self.batch_size:
                        yield torch.tensor(input_nids).to(self.device),torch.tensor(input_offset).to(self.device), torch.tensor(y).to(self.device), torch.tensor(clicks).to(self.device)

                        input_nids = []
                        input_offset = []
                        y = []
                        clicks = []


if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    device = torch.device('cuda')

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    EmbeddingSize = 64

    train, valid, test = utils.readdata(data_dir)

    data = DataLoader(train[:2000], device)

    model = DSSM(data.movie_count, EmbeddingSize).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    test_y = []
    test_y_ = []

    def fwbp():
        input_nids, input_offset, y, clicks = generator.next()

        #print x, y, clicks
        clicks_ = model.forward(input_nids, input_offset, y)

        #print clicks_, clicks
        loss = loss_fn(clicks_, clicks)
        loss.backward()

        #print clicks, clicks_, loss[0]
        del input_nids, input_offset, y, clicks
        return loss[0]

    easy_train.easy_train(fwbp, optimizer, data.train_iter_count)
    torch.save(model.input_embbag.state_dict(), model_save_path)

    nid_emb_fd = file('temp/dssm_out_emb.txt', 'w')
    easy_train.dump_embeddings(model.nid_emb, nid_emb_fd)


