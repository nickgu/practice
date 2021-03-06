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
        x = self.fc2(x)
        return x

if __name__=='__main__':
    iris_data = dataset.IrisData(label_onehot_encoding=False)
    X, Y = iris_data.data()
    X = X / Y.max()
    print X.shape
    print Y.shape


    x = Variable(torch.tensor(X)).float()
    y = Variable(torch.tensor(Y)).long()

    net = SoftmaxNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    #optimizer = optim.Adam(net.parameters(), lr=0.1)

    def feed_data():
        y_ = net.forward(x)
        loss = criterion(y_, y)
        loss.backward()
        return loss[0] 

    def single_feed_data():
        import random
        idx = random.randint(0, len(X)-1)
        
        x = Variable(torch.tensor([X[idx]])).float()
        y = Variable(torch.tensor([Y[idx]])).long()

        y_ = net.forward(x)
        loss = criterion(y_, y)
        loss.backward()
        return loss[0] 


    easy_train.easy_test(net, x, y)

    easy_train.easy_train(single_feed_data, optimizer, 10000)

    easy_train.easy_test(net, x, y)


