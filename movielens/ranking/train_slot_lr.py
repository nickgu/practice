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

class SlotLRRank(nn.Module):
    def __init__(self, user_count, item_count, user_genres_count, user_tags_count, embedding_size):
        # Ranking model:
        # input emb_size * 2 (embbag of input, emb of item to predict)
        # fc x 4
        nn.Module.__init__(self)
        
        pydev.info('user_count=%d' % user_count)
        pydev.info('item_count=%d' % item_count)
        pydev.info('user_genres_count=%d' % user_genres_count)
        pydev.info('user_tags_count=%d' % user_tags_count)
        pydev.info('embedding=%d' % embedding_size)

        self.uid_emb = nn.Embedding(user_count, embedding_size)
        self.iid_emb = nn.Embedding(item_count, embedding_size)
        self.user_genres_emb = nn.EmbeddingBag(user_genres_count, embedding_size, mode='mean')
        self.user_tags_emb = nn.EmbeddingBag(user_tags_count, embedding_size, mode='mean')

        self.lr = nn.Linear(embedding_size*3, 1)

    def forward(self, uid, iid, user_genres, user_genres_offset, user_tags, user_tags_offset):
        # sum up embeddings.
        user_emb = self.uid_emb(uid)
        item_emb = self.iid_emb(iid)
        user_genres_emb = self.user_genres_emb(user_genres, user_genres_offset)
        user_tags_emb = self.user_tags_emb(user_tags, user_tags_offset)

        x_ = torch.cat((user_emb, item_emb, user_genres_emb, user_tags_emb), 1)
        y = F.sigmoid( self.lr(x_) )
        return y

class DataGenerator:
    def __init__(self, train, device, epoch_count, batch_size):
        max_movie_id = 0
        max_user_id = 0

        pydev.info('load movies')
        self.movies = utils.load_movies('data/ml-20m_all', ignore_tags=False)

        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.data_count = 0
        self.device = device
        self.data = []

        write_progress = tqdm.tqdm(train)
        self.slot_coder = easy_train.SlotIndexCoder()
        # feature extracting.
        for uid, iid, click in write_progress:
            max_movie_id = max(max_movie_id, iid)
            max_user_id = max(max_user_id, uid)
            self.data_count += 1

            movie_id = int(iid)
            movie = self.movies.get(movie_id, utils.MovieInfo())
            user_genres = []
            for genres in movie.genres:
                key='%s_%s' % (uid, genres)
                idx = self.slot_coder.alloc('uid_genres', key)
                user_genres.append( idx )

            user_tags = []
            for tag_id, tag, score in movie.tags:
                if score < 0.1:
                    continue
                key='%s_%s' % (uid, tag)
                idx = self.slot_coder.alloc('uid_tag', key)
                #user_tags.append( idx )

            self.data.append( (uid, iid, user_genres, user_tags, click) )

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
            user_genres_ids = []
            user_genres_offset = []

            user_tags_ids = []
            user_tags_offset = []
            clicks = []

            epoch_num += 1
            pydev.info('Epoch %d' % epoch_num)
            for uid, iid, user_genres, user_tags, click in self.data:
                user_ids.append(uid)
                item_ids.append(iid)
                user_genres_offset.append( len(user_genres_ids) )
                user_genres_ids += user_genres

                user_tags_offset.append( len(user_tags_ids) )
                user_tags_ids += user_tags

                clicks.append(float(click))

                if len(clicks)>=self.batch_size:
                    yield (torch.tensor(user_ids).to(self.device),
                            torch.tensor(item_ids).to(self.device), 
                            torch.tensor(user_genres_ids).to(self.device), 
                            torch.tensor(user_genres_offset).to(self.device), 
                            torch.tensor(user_tags_ids).to(self.device), 
                            torch.tensor(user_tags_offset).to(self.device), 
                            torch.tensor(clicks).to(self.device))

                    user_ids = []
                    item_ids = []
                    clicks = []
                    user_genres_ids = []
                    user_genres_offset = []
                    user_tags_ids = []
                    user_tags_offset = []


if __name__=='__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage:\ndnn.py <datadir> <model>'
        sys.exit(-1)

    TestNum = -1
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

    pydev.info('save index (num=%d)' % len(data.idx_coder.tags))
    data.idx_coder.save(file('temp/user_genres.idx', 'w'))

    model = SlotLRRank(data.user_count, data.movie_count, len(data.idx_coder.tags), EmbeddingSize).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.BCELoss()
    
    generator = data.data_generator()

    test_y = []
    test_y_ = []

    def fwbp():
        user_ids, item_ids, user_genres_ids, user_genres_offset, user_tags_ids, user_tags_offset, clicks = generator.next()

        #print user_ids.size()
        #print item_ids.size()
        #print user_genres_ids.size()
        #print user_genres_offset.size()
        clicks_ = model.forward(user_ids, item_ids, user_genres_ids, user_genres_offset, user_tags_id, user_tags_offset)
        loss = loss_fn(clicks_, clicks)
        loss.backward()

        del user_ids, item_ids, user_genres_ids, user_genres_offset, user_tags_ids, user_tags_offset, clicks
        return loss[0]

    pydev.info('Begin training..')
    easy_train.easy_train(fwbp, optimizer, data.train_iter_count, loss_curve_output=file('log/train_loss.log', 'w'))


    pydev.info('Saving model..')
    torch.save(model.state_dict(), model_save_path)


