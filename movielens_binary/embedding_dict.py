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
        
class EmbeddingDict:
    def __init__(self, filename, seperator=',', contain_key=True, metric='angular'):
        '''
            File format(contain_key=True): 
               <key>[tab]<num>,<num>,... 

            File format(contain_key=False): 
               <num>,<num>,... 
        '''
        self.index = None # lazy create.
        self.emb_size = 0

        fd = file(filename)
        line_count = 0
        valid_count = 0
        for line in fd.readlines():
            if contain_key:
                key, value = line.strip().split('\t')
            else:
                key = line_count
                value = line.strip()

            line_count += 1
            value = value.split(',')

            d = len(value)
            if self.index is None:
                # first create.
                self.emb_size = d
                self.index = annoy.AnnoyIndex(f=self.emb_size, metric=metric)
                pydev.info('set emb_size=%d metric=%s' % (self.emb_size, metric))
            elif d != self.emb_size:
                continue

            vec = map(lambda x:float(x), value)
            self.index.add_item(int(key), vec)
            valid_count += 1
        
        pydev.info('emb load over, begin to build index..')
        self.index.build(32)
        pydev.info('EmbeddingDict load over: valid_count=%d, line_count=%d' %(valid_count, line_count))
        

if __name__=='__main__':
    filename = sys.argv[1]
    index = EmbeddingDict(filename, contain_key=False, metric='dot')

    # load movielens movie_info.
    movie_info = {}
    for row in pydev.foreach_row(file('data/ml-20m/movies.csv'), seperator=','):
        movie_id = row[0]
        genres = row[-1]
        title = ','.join(row[1:-1])
        movie_info[movie_id] = title+' : '+genres

    while True:
        try:
            sys.stdout.write('Query: ')
            query_id = input()
            query = str(query_id)

            print '%s: %s' % (query, movie_info.get(query,'not_found.')) 

            print '--- Search Result ---'
            ans, dis = index.index.get_nns_by_item(query_id, n=30, include_distances=True)
            for item, score in zip(ans, dis):
                item = str(item)
                print '%s [%.3f] %s' %(item, score, movie_info.get(item,'none.'))
            print 
        except:
            pydev.err('runtime error..')
