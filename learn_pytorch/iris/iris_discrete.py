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

class SoftmaxNet(nn.Module):
    def __init__(self):
        super(SoftmaxNet, self).__init__()
        self.fc1 = nn.Linear(400, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x0 = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x0) )
        x = F.relu( self.fc3(x) )
        x = self.fc4(x + x0)
        return x

if __name__=='__main__':
    iris_data = dataset.IrisData(label_onehot_encoding=False)
    X, Y = iris_data.data()


    new_X = np.zeros( (X.shape[0], 400) )
    for vi, v in enumerate(X):
        for i, z in enumerate(v):
            idx = i * 100 + int(z*10)
            new_X[vi][idx] = 1

    print new_X.shape
    print Y.shape


    net = SoftmaxNet()
    criterion = nn.CrossEntropyLoss()

    optimizer_sgd01 = optim.SGD(net.parameters(), lr=0.1)
    optimizer_sgd001 = optim.SGD(net.parameters(), lr=0.01)
    optimizer_adam = optim.Adam(net.parameters(), lr=0.1)

    dtype = torch.FloatTensor
    loader = torch.utils.data.DataLoader(zip(new_X, Y), shuffle=True, batch_size=150)

    epoch_num = 0
    best_hit = 0
    best_ps = ''
    optimizer = optimizer_sgd01
    while 1:
        print >> sys.stderr, 'input n epoch to run..'
        l=sys.stdin.readline()
        try:
            if ',' in l:
                l, op = l.strip().split(',')
                if op == 'adam':
                    optimizer = optimizer_adam
                elif op == 'sgd001':
                    optimizer = optimizer_sgd001
                elif op == 'sgd01':
                    optimizer = optimizer_sgd01
            n = int(l)
        except:
            break

        print optimizer

        for i in range(n):
            epoch_num += 1
            
            data_iter = iter(loader)
            run_loss = 0
            right = 0
            total = 0
            for t, (x, y) in enumerate(data_iter):
                inputs = Variable(x).type(dtype)
                labels = Variable(y).type(torch.LongTensor)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                # temp test precision
                # seems different max need keepdim..
                #pred = outputs.max(1, keepdim=True)[1]
                pred = outputs.max(1)[1]

                p = pred.eq(labels.view_as(pred)).sum()
                right += p.data[0]
                total += len(pred)
                precision = right * 100.0 / total

                loss = criterion(outputs, labels)
                run_loss += loss.data[0]
                loss.backward()
                optimizer.step()

            # print statistics
            ops = 'p=%.2f%% (%d/%d)' % (precision, right, total)
            ps = ops
            if right >= 145:
                ps = pydev.ColorString.green(ops)
            if right >= 147:
                ps = pydev.ColorString.cyan(ops)
            if right >= 148:
                ps = pydev.ColorString.red(ops)
            if right >= 149:
                ps = pydev.ColorString.yellow(ops)

            if right > best_hit:
                best_hit = right
                best_ps = ps
   
            print('[%d, %d/%d] loss: %.5f %s curbest=%s' %
              (epoch_num, i, n, run_loss, ps, best_ps))
            if best_hit == 150:
                # we found the best answers!.
                break

        if best_hit == 150:
            # we found the best answers!.
            break

    
