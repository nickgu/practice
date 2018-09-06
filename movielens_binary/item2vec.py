#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import annoy
import pydev

class ItemIndex:
    def __init__(self, filename, embedding_size=100):
        self.index = annoy.AnnoyIndex(f=embedding_size, metric='dot')

        fd = file(filename)
        for line in fd.readlines():
            row = line.strip().split(' ')
            if len(row)!=embedding_size+1:
                continue
            key = row[0]
            if key == '':
                continue

            key = int(key)
            vec = map(lambda x:float(x), row[1:])

            self.index.add_item(key, vec)
        
        self.index.build(32)
        

if __name__=='__main__':
    import numpy as np
    from scipy import spatial

    fd = file(sys.argv[1])
    vec_dict = {}
    for line in fd.readlines():
        row = line.strip().split(' ')
        if len(row)!=101:
            continue
        key = row[0]
        if key == '':
            continue

        key = int(key)
        vec = map(lambda x:float(x), row[1:])

        vec_dict[key] = np.array(vec)
    print >> sys.stderr, 'load ok!'

    while 1:
        line = sys.stdin.readline()
        a, b = line.strip().split(',')
        print a, b

        try:
            va = vec_dict[int(a)]
            vb = vec_dict[int(b)]
            print 'dot: %.3f' % (va.dot(vb))
            print 'cos: %.3f' % (1.0 - spatial.distance.cosine(va, vb))

        except:
            print 'Error!'
            continue

        

