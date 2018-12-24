#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import random
import pydev

class MovieLensRatingsBinaryReader:
    def __init__(self, stream):
        self.data = []
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

            self.data.append( (uid, iid, score) )

    def sample_train_validation_and_test(self, validation_size = 500000, test_size = 500000):

        # shuffle data.
        random.shuffle(self.data)

        # |<-- validataion -->|<-- test -->|<-- data -->|
        validation = self.data[:validation_size]
        test = self.data[validation_size:validation_size + test_size]
        data = self.data[validation_size + test_size :]

        return data, validation, test


if __name__=='__main__':
    # Movielens Data Generator.
    # generate datasets from rating.csv
    #   
    #   ml_binary_reader.py rating.csv output_data
    #
    if len(sys.argv)!=3:
        print >> sys.stderr, 'ml_binary_reader.py rating.csv output_dir'
        sys.exit(-1)

    fd = file(sys.argv[1])
    reader = MovieLensRatingsBinaryReader(fd)
    print >> sys.stderr, 'Load over data=%d' % (len(reader.data))
    train, validation, test = reader.sample_train_validation_and_test()
    print >> sys.stderr, 'Genrate set over, begin dumpping.'

    def write(data, fd):
        for uid, iid, score in data:
            print >> fd, '%s,%s,%s' % (uid, iid, score)
    
    write(train, file(sys.argv[2] + '/train', 'w'))
    write(validation, file(sys.argv[2] + '/valid', 'w'))
    write(test, file(sys.argv[2] + '/test', 'w'))




