#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import load_data
import pydev
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import *

#import models


class ConvBNReluPool(nn.Module):
    def __init__(self, in_chan, out_chan, pool=True):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, [3,3], padding=1),
                nn.BatchNorm2d(out_chan)
                )
        self.maxpool = pool

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = F.relu(x)
        if self.maxpool:
            x = F.max_pool2d(x, [2,2])
        return x


class ResConvBlock(nn.Module):
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
        x = input
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        return input + x


class TempModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
                ConvBNReluPool(3, 64),
                ConvBNReluPool(64, 128, pool=True),
                ResConvBlock(128),
                ConvBNReluPool(128, 256, pool=True),
                ConvBNReluPool(256, 512),
                ResConvBlock(512)
                )
        self.fc = nn.Linear(2048, 10)

    def forward(self, input):
        x = input
        x = self.net(x)
        x = x.view((-1,2048))
        x = self.fc(x)
        return x

if __name__=='__main__':
    arg = pydev.Arg('Cifar10 training program with pytorch.')
    arg.str_opt('batch', 'b', 'batch size', default='512')
    arg.str_opt('epoch', 'e', 'epoch count', default='100')
    arg.str_opt('step', 's', 'step count', default='3000')
    opt = arg.init_arg()

    step_size = int(opt.step)
    print 'step: ', step_size
    
    epoch = int(opt.epoch)
    print 'epoch: ', epoch

    batch_size = int(opt.batch)
    print 'batch_size: ', batch_size

    # make simple Model.

    train_transform = Compose([
        RandomCrop(32, padding=2), 
        RandomHorizontalFlip(),
        #RandomRotation(30),
        ToTensor(),
        Normalize(mean=(125.31, 122.95, 113.87), std=(62.99, 62.09, 66.70))
        ])
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=(125.31, 122.95, 113.87), std=(62.99, 62.09, 66.70))
        ])

    train = torchvision.datasets.cifar.CIFAR10('../../dataset/', transform=train_transform)
    test =  torchvision.datasets.cifar.CIFAR10('../../dataset/', train=False, transform=test_transform)

    # train phase.
    #model = models.SimpleConvNet()
    #model = models.Stack5ConvNet()
    #model = models.Res9Net()
    
    model = TempModel()

    sys.path.append('../')
    import easy_train

    cuda = torch.device('cuda')     # Default CUDA device
    model.to(cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.NLLLoss()

    easy_train.epoch_train(train, model, optimizer, loss_fn, epoch, 
            batch_size=batch_size, device=cuda, validation=test, validation_epoch=3)
    easy_train.epoch_test(test, model, device=cuda)

    print 'train over'
    #easy_train.easy_test(model, train_x, train_y)
    #easy_train.easy_test(model, test_x, test_y)



