#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 
# this program learn and generate 
# chinese peotry by gru.
#
# dataset from: 
#   https://github.com/nickgu/chinese-poetry
#

import os
import json
import sys
import random

import gru


class PoetryRepo:
    def __init__(self, path='dataset/poetry'):
        self.__all_poetry = []
        for filename in os.listdir(path):
            if filename.startswith('poet.'):
                with file(path + '/' + filename) as fd:
                    d = json.loads(fd.read())
                    for item in d:
                        if 'paragraphs' in item:
                            self.__all_poetry.append( u''.join(item['paragraphs']) )
        
        print >> sys.stderr, 'load over [%d poets loaded.]' % len(self.__all_poetry)
        print >> sys.stderr, '%s' % self.__all_poetry[random.randint(0, len(self.__all_poetry)-1)].encode('utf-8')


    def __iter__(self):
        for item in self.__all_poetry:
            yield item

if __name__=='__main__':
    lang = gru.Lang('poetry')
    poetry_repo = PoetryRepo()

    datas = []

    for poet in poetry_repo:
        vec = lang.addSentence(poet)
        datas.append(vec)

    print len(datas)
