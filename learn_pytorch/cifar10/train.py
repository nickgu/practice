#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import load_data
import pydev
import sys

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import *

import models

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

    transform_ops = Compose([
        RandomCrop(32, padding=2), 
        RandomHorizontalFlip(),
        #RandomRotation(30),
        ToTensor()])
    train = torchvision.datasets.cifar.CIFAR10('../../dataset/', transform=transform_ops)
    test =  torchvision.datasets.cifar.CIFAR10('../../dataset/', train=False, transform=transform_ops)

    # train phase.
    #model = models.SimpleConvNet()
    #model = models.Stack5ConvNet()
    model = models.Res9Net()

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



