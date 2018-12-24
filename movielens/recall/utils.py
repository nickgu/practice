#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import tqdm

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
    print >> sys.stderr, 'load [%s/]' % dir
    train = readfile(file(dir + '/train'), test_num)
    valid = readfile(file(dir + '/valid'))
    test = readfile(file(dir + '/test'))
    print >> sys.stderr, 'load over'
    return train, valid, test

def measure(predictor, test, debug=False):
    P = []
    R = []
    total_answers = 0
    generator = test
    if debug:
        generator = test[:1]
    progress = tqdm.tqdm(generator)
    for uid, items in progress:
        m = len(items) / 2
        input = items[:m]
        ans = filter(lambda x:x[1]==1, items[m:])

        if debug:
            print '--- User [%s] ---' % uid
            print 'input: [%s]' % (','.join(map(lambda x:'%s:%s'%(x[0], x[1]), input)))
            print 'expect: [%s]' % (','.join(map(lambda x:'%s:%s'%(x[0], x[1]), ans)))

        output = predictor(uid, input)
        total_answers += len(output)

        if debug:
            print 'output: [%s]' % (','.join(output))

        if len(ans) == 0:
            continue
        if len(output) == 0:
            continue

        hit = len(set(output).intersection( set(map(lambda x:x[0], ans)) ))

        P.append( hit * 1.0 / len(output) )
        R.append( hit * 1.0 / len(ans) )

    precision = sum(P) * 100.0 / len(P)
    recall = sum(R) * 100.0 / len(R)
    print 'answers per user: %.2f' % (total_answers * 1.0 / len(test)) 
    print 'P : %.2f%%' % precision
    print 'R : %.2f%%' % recall
    print 'F : %.2f%%' % (precision*recall*2. / (precision + recall))


if __name__=='__main__':
    train, valid, test = readdata(sys.argv[1])
    print len(train)
    print len(valid)
    print len(test)
