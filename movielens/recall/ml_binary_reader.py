#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import random
import pydev

class MovieLensRatingsBinaryReader:
    def __init__(self, stream):
        self.__users = {}

        cur_uid = None
        cur_queue = []
        for row in pydev.foreach_row(stream, seperator=',', min_fields_num=4):
            uid, iid, rating, ts = row
            if uid == 'userId':
                continue
            
            uid = int(uid)
            iid = int(iid)
            rating = float(rating)
            # make it binary.
            score = 0
            if rating >= 4:
                score = 1
            ts = int(ts)

            if uid != cur_uid:
                # sorted by ts.
                if len(cur_queue)>0:
                    cur_queue = sorted(cur_queue, key=lambda x:x[2])
                    self.__users[cur_uid] = cur_queue
                
                cur_uid = uid
                cur_queue = []

            cur_queue.append( (iid, score, ts) )

        cur_queue = sorted(cur_queue, key=lambda x:x[2])
        self.__users[cur_uid] = cur_queue

    def user_actions(self):
        for user, actions in self.__users.iteritems():
            yield user, actions


    def sample_train_validation_and_test(self, validation_size = 1000, test_size = 1000):
        data = [(uid, actions) for uid, actions in self.__users.iteritems()]
        
        validation = random.sample(data, validation_size)
        vset = set(map(lambda x:x[0], validation))
        data = filter(lambda x:x[0] not in vset, data)

        test = random.sample(data, test_size)
        tset = set(map(lambda x:x[0], test))
        data = filter(lambda x:x[0] not in tset, data)

        return data, validation, test


if __name__=='__main__':
    # generate datasets from rating.csv
    #   
    #   ml_binary_reader.py rating.csv output_data
    #
    if len(sys.argv)!=3:
        print >> sys.stderr, 'ml_binary_reader.py rating.csv output_dir'
        sys.exit(-1)

    fd = file(sys.argv[1])
    reader = MovieLensRatingsBinaryReader(fd)
    print >> sys.stderr, 'Load over'
    train, validation, test = reader.sample_train_validation_and_test()
    print >> sys.stderr, 'Genrate set over, begin dumpping.'

    
    def write(data, fd):
        for uid, actions in data:
            print >> fd, '%s\t%s' % (uid, ','.join(map(lambda x:'%s:%s:%s'%(x[0],x[1],x[2]), actions)))
    
    write(train, file(sys.argv[2] + '/train', 'w'))
    write(validation, file(sys.argv[2] + '/valid', 'w'))
    write(test, file(sys.argv[2] + '/test', 'w'))




