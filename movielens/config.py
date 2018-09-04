#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

(D_100k, D_1m) = range(2)
DataType = D_1m

if DataType == D_100k:
    DataDir = 'ml-100k'
    Seperator = '\t'
    PropertySeperator = '|'
    TrainData = 'ua.base'
    TestData = 'ua.test'
    ItemData = 'u.item'
    UserData = 'u.user'
    
elif DataType == D_1m:
    DataDir = 'ml-1m'
    Seperator = '::'
    PropertySeperator = '::'
    TrainData = 'train.dat'
    TestData = 'test.dat'
    ItemData = 'movies.dat'
    UserData = 'users.dat'

if __name__=='__main__':
    pass
