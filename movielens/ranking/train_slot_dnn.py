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
import easy.slot_file
import easy.pytorch
import utils


import tqdm
import numpy as np

class SlotDnnRank(nn.Module):
    def __init__(self, slot_info, embedding_size, device):
        # Ranking model:
        # input emb_size * N
        # fc x 4
        nn.Module.__init__(self)
        
        total_input_length = 0
        self.emb_bags = []
        for slot, slot_feanum in slot_info:
            pydev.info('init embeding bag of %s (%d)' % (slot, slot_feanum))
            self.emb_bags.append( nn.EmbeddingBag(slot_feanum, embedding_size, mode='mean').to(device) )
            total_input_length += embedding_size

        self.fc1 = nn.Linear(total_input_length, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # sum up embeddings.
        x_emb = []
        #print x
        for idx, bag in enumerate(self.emb_bags):
            e = bag(x[idx][0], x[idx][1])
            x_emb.append(e)

        #print x_emb
        x_ = torch.cat(x_emb, 1)
        x_ = F.relu(self.fc1(x_))
        x_ = F.relu(self.fc2(x_))
        y = F.sigmoid( self.out(x_) )

        return y


if __name__=='__main__':
    autoarg = pydev.AutoArg()

    TestNum = -1
    EmbeddingSize = int(autoarg.option('emb', 32))
    EpochCount = int(autoarg.option('epoch', 4))
    BatchSize = int(autoarg.option('batch', 10000))
    device_name = autoarg.option('device', 'cuda')
    input_filename = autoarg.option('f')
    slotinfo_filename = autoarg.option('s')
    model_save_path = autoarg.option('o')

    device = torch.device(device_name)

    reader = easy.slot_file.SlotFileReader(input_filename)
    
    # temp get slot_info.
    slot_info = []
    for slot, slot_feanum in pydev.foreach_row(file(slotinfo_filename),format='si'):
        slot_info.append( (slot, slot_feanum) )

    model = SlotDnnRank(slot_info, EmbeddingSize, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.BCELoss()
    
    def fwbp():
        labels, slots = reader.next(BatchSize)
        
        # make pytorch data.
        clicks = torch.Tensor(labels).to(device)
        dct = {}
        for item in slots:
            for slot, ids in item:
                if slot not in dct:
                    # id_list, offset
                    dct[slot] = [[], []]

                lst = dct[slot][0]
                idx = dct[slot][1]
                idx.append( len(lst) )
                lst += ids

        x = []
        for slot, _ in slot_info:
            id_list, offset = dct.get(slot, [[], []])
            emb_pair = torch.tensor(id_list).to(device), torch.tensor(offset).to(device)
            x.append(emb_pair)

        clicks_ = model.forward(x)
        loss = loss_fn(clicks_, clicks)
        loss.backward()

        del x, clicks
        return loss[0]

    def while_condition():
        return reader.epoch() < EpochCount

    pydev.info('Begin training..')
    easy.pytorch.common_train(fwbp, optimizer, -1, while_condition=while_condition, loss_curve_output=file('log/train_loss.log', 'w'))

    pydev.info('Saving model..')
    torch.save(model.state_dict(), model_save_path)


