#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import load_data
import pydev

if __name__=='__main__':
    arg = pydev.Arg('Cifar10 training program with pytorch.')
    opt = arg.init_arg()

    #train_x, train_y = load_data.load_all_data()
    train_x, train_y = load_data.load_one_part()
    test_x, test_y = load_data.load_test()

