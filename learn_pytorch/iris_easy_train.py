#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import numpy as np
import pydev

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import sys
sys.path.append('../')

import dataset

import easy_train

class SoftmaxNet(nn.Module):
    def __init__(self):
        super(SoftmaxNet, self).__init__()
        self.fc1 = nn.Linear(4, 1024)
        self.fch = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 3)
        

    def forward(self, x):
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fch(x) )
        x = F.softmax( self.fc2(x) )
        return x

if __name__=='__main__':
    iris_data = dataset.IrisData(label_onehot_encoding=False)
    x, y = iris_data.data()
    x = x / y.max()
    print x.shape
    print y.shape


    x = Variable(torch.tensor(x)).float()
    y = Variable(torch.tensor(y)).long()

    net = SoftmaxNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    #optimizer = optim.Adam(net.parameters(), lr=0.1)


    def feed_data():
        y_ = net.forward(x)
        loss = criterion(y_, y)
        loss.backward()
        return loss[0] 

    easy_train.easy_test(net, x, y)

    easy_train.easy_train(feed_data, None, optimizer, iteration_count=10000)

    easy_train.easy_test(net, x, y)


