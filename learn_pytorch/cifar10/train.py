#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import load_data
import pydev
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import *

import models


class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        bx = random.randint(0, x.shape[0]-1)
        by = random.randint(0, x.shape[1]-1)
        x[..., bx:bx+self.size, by:by+self.size ] = 0.
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
        RandomCrop(32, padding=4, padding_mode='reflect'), 
        RandomHorizontalFlip(),
        ToTensor(),
        #Cutout(8),
        RandomErasing(p=0.7, scale=(0.2, 0.2), value='random'),
        Normalize(mean=(125.31, 122.95, 113.87), std=(62.99, 62.09, 66.70)),
        ])
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=(125.31, 122.95, 113.87), std=(62.99, 62.09, 66.70)),
        ])

    train = torchvision.datasets.cifar.CIFAR10('../../dataset/', transform=train_transform)
    #test =  torchvision.datasets.cifar.CIFAR10('../../dataset/', train=False, transform=train_transform)
    test =  torchvision.datasets.cifar.CIFAR10('../../dataset/', train=False, transform=test_transform)

    # train phase.
    #model = models.SimpleConvNet()
    #model = models.Stack5ConvNet()
    #model = models.Res9Net()
    #model = models.TempModel()
    model = models.V3_ResNet()

    sys.path.append('../')
    import easy_train

    cuda = torch.device('cuda')     # Default CUDA device
    model.to(cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    '''
    def lr_scheduler(cur):
        if cur < 5:
            return (cur+1) / 5. * 0.1
        elif cur < 10:
            return 0.1 - (cur-5.) / 5. * 0.1 + 0.1
        elif cur < 15:
            return 0.01 - (cur - 10.) / 5. * 0.009
        else:
            return 1e-1
    '''
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.05)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, verbose=True, factor=0.5)

    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.NLLLoss()

    easy_train.epoch_train(train, model, optimizer, loss_fn, epoch, 
            batch_size=batch_size, device=cuda, validation=test, validation_epoch=3, 
            scheduler=None)
            #scheduler=scheduler)
            #validation_scheduler=scheduler)
    easy_train.epoch_test(test, model, device=cuda)

    print 'train over'
    #easy_train.easy_test(model, train_x, train_y)
    #easy_train.easy_test(model, test_x, test_y)



