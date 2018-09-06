#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import torch
import torch.nn as nn
import utils

class UserAutoEncoder(nn.Model):
    def __init__(self, input_size, embedding_size):
        self.fc = nn.linear(input_size, embedding_size)
        self.fc2 = nn.linear(embedding_size, input_size)
        self.loss = #

    def forward(self, x):
        emb = F.relu( self.fc(x) )
        y = F.relu( self.fc2(emb) )
        loss = self.loss(x, y)

if __name__=='__main__':
    train, valid, test = utils.readdata(sys.argv[1])


