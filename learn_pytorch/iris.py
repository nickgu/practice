#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('../')

import dataset

class SoftmaxNet(nn.Module):
    def __init__(self):
        super(SoftmaxNet, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return F.softmax(  self.fc(x) )

if __name__=='__main__':
    iris_data = dataset.IrisData(label_onehot_encoding=False)
    X, Y = iris_data.data()
    print X.shape
    print Y.shape


    net = SoftmaxNet()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.2)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    dtype = torch.FloatTensor
    inputs = torch.from_numpy(X)
    labels = torch.from_numpy(Y)
    inputs = Variable(inputs).type(dtype)
    labels = Variable(labels).type(torch.LongTensor)

    for t in range(500):  # loop over the dataset multiple times
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # temp test precision
        pred = outputs.max(1, keepdim=True)[1]
        p = pred.eq(labels.view_as(pred)).sum()
        prec = 100. * p.data[0] / 150.

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print('[iter = %d] loss: %.3f p=%.2f%% (%d)' %
              (t, loss.data[0], prec, p.data[0]))
