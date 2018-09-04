#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
from torch.autograd import Variable

if __name__=='__main__':
    # forward.
    x = Variable(torch.ones(5), requires_grad=True)
    y = x * x + 2 * x + 1
    z = y.mean()
    print z
    
    # backward.
    z.backward()

    print x.grad
