#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import random
import pydev

class MovieLensRatingsReader:
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
    #   movielens_binary_reader.py rating.csv
    #
    pass
