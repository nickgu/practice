#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys

def readfile(fd, test_num=-1):
    data = []
    for line in fd.readlines():
        uid, items = line.split('\t')
        d = map(lambda x:x.split(':'), items.split(','))
        d = map(lambda x:(x[0], int(x[1]), int(x[2])), d)
        data.append( (uid, d))
        if test_num>0 and len(data)>=test_num:
            break
    return data

def readdata(dir, test_num=-1):
    train = readfile(file(dir + '/train'), test_num)
    valid = readfile(file(dir + '/valid'))
    test = readfile(file(dir + '/test'))
    return train, valid, test

def measure(predictor, test):
    total_hit = 0
    total_output = 0
    total_ans = 0
    for uid, items in test:
        items = filter(lambda x:x[1]==1, items)

        m = len(items) / 2
        input = items[:m]
        output = predictor(uid, input)
        ans = items[m:]

        hit = len(set(output).intersection( set(map(lambda x:x[0], ans)) ))
        total_hit += hit
        total_output += len(output)
        total_ans += len(ans)

    print 'P : %.2f%%' % (total_hit * 100.0 / total_output)
    print 'R : %.2f%%' % (total_hit * 100.0 / total_ans)

if __name__=='__main__':
    train, valid, test = readdata(sys.argv[1])
    print len(train)
    print len(valid)
    print len(test)
