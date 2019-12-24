#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
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

class ConvBNResPool(nn.Module):
    def __init__(self, in_chan, out_chan):
        nn.Module.__init__(self)
        
        self.conv_bn = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, [3,3], padding=1),
                nn.BatchNorm2d(out_chan)
            )

    def forward(self, input):
        x = input
        #print 'conv_bn:'
        #print input.shape
        x = self.conv_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print x.shape
        return x


class VGGBlock(nn.Module):
    def __init__(self, in_chan, out_chan, max_pool=True):
        nn.Module.__init__(self)
        self.convs = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, [3,3], padding=1),
                nn.Conv2d(out_chan, out_chan, [3,3], padding=1),
                nn.Conv2d(out_chan, out_chan, [3,3], padding=1)
            )
        self.max_pool = max_pool

    def forward(self, input):
        x = self.convs(input)
        if self.max_pool:
            x = F.max_pool2d(x, [2,2])
        return x

class ConvResBlock(nn.Module):
    def __init__(self, chan):
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
                nn.Conv2d(chan, chan, [3,3], padding=1),
                nn.BatchNorm2d(chan)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(chan, chan, [3,3], padding=1),
                nn.BatchNorm2d(chan)
            )

    def forward(self, input):
        #print 'conv_res:'
        #print input.shape

        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #print x.shape
        return x+input

class SimpleConvNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.convs = nn.Sequential(
            Conv2DPool(3, 64, [5,5], [2,2], padding=2),
            Conv2DPool(64, 32, [3,3], [2,2], padding=1),
            Conv2DPool(32, 32, [3,3], [2,2], padding=1)
            )

        self.fc = nn.Linear(512, 10)

    def forward(self, input):
        x = input
        x = self.convs(x)
        # flatten
        x = x.view( x.shape[0], -1 )
        #print x.shape
        y = self.fc(x)
        return y

class Stack5ConvNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.convs = nn.Sequential(
                VGGBlock(3, 64),
                VGGBlock(64, 128),
                VGGBlock(128, 256)
            )

        self.fc = nn.Linear(4096, 10)

    def forward(self, input):
        x = input
        x = self.convs(x)
        # flatten
        x = x.view( x.shape[0], -1 )
        y = self.fc(x)
        return y


class Res9Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.convs = nn.Sequential(
                ConvBNResPool(3, 64),
                ConvBNResPool(64, 128),
                ConvResBlock(128),
                ConvBNResPool(128, 256),
                ConvResBlock(256)
            )

        self.fc = nn.Linear(4096, 10)

    def forward(self, input):
        x = input
        x = self.convs(x)
        # flatten
        x = x.view( x.shape[0], -1 )
        y = self.fc(x)
        return y



