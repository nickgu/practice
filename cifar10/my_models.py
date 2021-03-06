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



class ConvBNReluPool(nn.Module):
    def __init__(self, in_chan, out_chan, pool=False):
        nn.Module.__init__(self)
        if pool:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, [3,3], padding=1, bias=False),
                    nn.BatchNorm2d(out_chan),
                    nn.MaxPool2d(2)
                    )
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, [3,3], padding=1, bias=False),
                    nn.BatchNorm2d(out_chan)
                    )

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = F.relu(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(self, chan):
        nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
                nn.Conv2d(chan, chan, [3,3], padding=1, bias=False),
                nn.BatchNorm2d(chan)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(chan, chan, [3,3], padding=1, bias=False),
                nn.BatchNorm2d(chan)
                )


    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        return input + x


class V2_ResNet(nn.Module):
    '''
        92.29% at batch=128
    '''
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
                ConvBNReluPool(3, 64),
                ConvBNReluPool(64, 128, pool=True),
                ResConvBlock(128),
                ConvBNReluPool(128, 256, pool=True),
                ConvBNReluPool(256, 512, pool=True),
                ResConvBlock(512),
                nn.MaxPool2d(4)
                )
        self.fc = nn.Linear(512, 10, bias=False)

    def forward(self, input):
        x = input
        x = self.net(x)
        x = x.view((-1,512))
        x = self.fc(x)
        return x

class V3_ResNet(nn.Module):
    '''
        90% at E32

        V0: augment
        train_transform = Compose([
        RandomCrop(32, padding=4, padding_mode='reflect'), 
        RandomHorizontalFlip(),
        ToTensor(),
        #Cutout(8),
        RandomErasing(p=0.5, scale=(0.1, 0.1)), #, value='random'),
        Normalize(mean=(125.31, 122.95, 113.87), std=(62.99, 62.09, 66.70)),
        ])

    '''
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
                ConvBNReluPool(3, 64),
                ConvBNReluPool(64, 128, pool=True),
                ResConvBlock(128),
                ConvBNReluPool(128, 256, pool=True),
                ConvBNReluPool(256, 512, pool=True),
                ConvBNReluPool(512, 1024, pool=True),
                ResConvBlock(1024),
                nn.MaxPool2d(2)
                )
        self.fc = nn.Linear(1024, 10, bias=False)

    def forward(self, input):
        x = input
        x = self.net(x)
        x = x.view((-1,1024))
        x = self.fc(x)
        return x

class V4_ResNet(nn.Module):
    '''
        90% at E68 (bs=128)
        90% at E26 (V0_augment), 300~92.56
    '''
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
                ConvBNReluPool(3, 64),
                ConvBNReluPool(64, 128, pool=True),
                ResConvBlock(128),
                ConvBNReluPool(128, 256, pool=True),
                ConvBNReluPool(256, 512, pool=True),
                ConvBNReluPool(512, 1024, pool=True),
                ResConvBlock(1024),
                ConvBNReluPool(1024, 2048, pool=True),
                )
        self.fc = nn.Linear(2048, 10, bias=False)

    def forward(self, input):
        x = input
        x = self.net(x)
        x = x.view((-1,2048))
        x = self.fc(x)
        return x


