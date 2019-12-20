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
    arg.str_opt('step', 's', 'step count', default='3000')
    opt = arg.init_arg()

    step_size = int(opt.step)
    print 'step: ', step_size

    #train_x, train_y = load_data.load_all_data()
    train_x, train_y = load_data.load_one_part()
    test_x, test_y = load_data.load_test()

    # make simple Model.
    
    train_x = torch.tensor(train_x).float() / 256.0
    train_y = torch.tensor(train_y).long()
    test_x = torch.tensor(test_x).float() / 256.0
    test_y = torch.tensor(test_y).long()

    '''
    print train_x.shape
    train_x = train_x.transpose(3,1)
    print 'after transpose', train_x.shape
    '''

    '''
    print test_x.shape
    test_x = test_x.transpose(3,1)
    print test_x.shape
    '''

    # train phase.
    model = Cifar10Network()

    sys.path.append('../')
    import easy_train


    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32
    loader_iter = iter(torch.utils.data.DataLoader(zip(train_x, train_y), batch_size = batch_size))

    def forward_and_backward():
        x, y = loader_iter.next()

        y_ = model.forward(x)
        loss = loss_fn(y_, y)
        loss.backward()
        return loss.item() / batch_size

    easy_train.easy_train(forward_and_backward, optimizer, step_size)

    easy_train.easy_test(model, test_x, test_y)



