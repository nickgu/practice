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

'''
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
'''

class Cifar10Network(nn.Module):
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



if __name__=='__main__':
    arg = pydev.Arg('Cifar10 training program with pytorch.')
    arg.str_opt('batch', 'b', 'batch size', default='32')
    arg.str_opt('step', 's', 'step count', default='3000')
    arg.str_opt('epoch', 'e', 'epoch count', default='200')
    opt = arg.init_arg()

    step_size = int(opt.step)
    print 'step: ', step_size
    
    epoch = int(opt.epoch)
    print 'epoch: ', epoch

    batch_size = int(opt.batch)
    print 'batch_size: ', batch_size

    # make simple Model.

    transform_ops = Compose([
        RandomCrop(32, padding=2), 
        RandomHorizontalFlip(),
        RandomRotation(30),
        ToTensor()])
    train = torchvision.datasets.cifar.CIFAR10('../../dataset/', transform=transform_ops)
    test =  torchvision.datasets.cifar.CIFAR10('../../dataset/', train=False, transform=transform_ops)

    # train phase.
    model = Cifar10Network()

    sys.path.append('../')
    import easy_train

    cuda = torch.device('cuda')     # Default CUDA device
    model.to(cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    easy_train.epoch_train(train, model, optimizer, batch_size=batch_size,
            loss_fn, epoch, device=cuda, validation=test, validation_epoch=5)
    easy_train.epoch_test(test, model, device=cuda)

    print 'train over'
    #easy_train.easy_test(model, train_x, train_y)
    #easy_train.easy_test(model, test_x, test_y)



