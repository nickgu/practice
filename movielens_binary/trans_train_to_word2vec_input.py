#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils

if __name__=='__main__':
    output_file = file('temp/word2vec.input', 'w')
    for uid, items in pydev.foreach_row(file('data/train')):
        actions = []
        for item in items.split(','):
            vals = item.split(':')
            if vals[1] == '0':
                continue

            actions.append(vals[0])

        print >> output_file, ' '.join(actions)
