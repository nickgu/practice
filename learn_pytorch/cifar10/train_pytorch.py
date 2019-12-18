#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import load_data
import pydev

import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Conv2DPool(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pooling_kernel, stride=1, padding=0):
        nn.Module.__init__(self)
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
        self.pool_kernel = pooling_kernel

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = F.max_pool2d(x, self.pool_kernel)
        return x

class FCNetworkStack(nn.Module):
    def __init__(self, stack_width, active=F.relu):
        nn.Module.__init__(self)
        
        self.layers = []
        self.active = active
        for idx in range(len(stack_width)-1):
            n_in = stack_width[idx]
            n_out = stack_width[idx+1]
            
            layer = nn.Linear(n_in, n_out)
            self.layers.append(layer)

    def forward(self, input):
        for l in self.layers:
            input = l(input)
            input = self.active(input)
            #print input
        return input


class Cifar10Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.convs = []
        self.convs.append( Conv2DPool(3, 64, [5,5], [2,2], padding=2) )
        self.convs.append( Conv2DPool(64, 32, [3,3], [2,2], padding=1) )
        self.convs.append( Conv2DPool(32, 32, [3,3], [2,2], padding=1) )

        self.stack_fc = FCNetworkStack([512, 256, 128, 10])
        self.softmax = F.softmax

    def forward(self, input):
        x = input
        for conv in self.convs:
            x = conv(x)

        x = x.view( x.shape[0], -1 )
        #print x.shape
        x = self.stack_fc(x)
        y = self.softmax(x)
        return y

if __name__=='__main__':
    arg = pydev.Arg('Cifar10 training program with pytorch.')
    opt = arg.init_arg()

    #train_x, train_y = load_data.load_all_data()
    train_x, train_y = load_data.load_one_part()
    test_x, test_y = load_data.load_test()

    # make simple Model.
    
    train_x = T.tensor(train_x).float() / 256.0
    print train_x.shape
    train_x = train_x.transpose(3,1)
    print 'after transpose', train_x.shape

    test_x = T.tensor(train_x).float() / 256.0
    print test_x.shape
    test_x = test_x.transpose(3,1)
    print test_x.shape

    t = test_x[:50]
    model = Cifar10Network()
    print model(test_x)




