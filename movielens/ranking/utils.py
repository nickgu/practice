#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import tqdm

import pydev

import sklearn
from sklearn import metrics

class MovieInfo:
    def __init__(self):
        self.id = None
        self.title = None
        self.tags = []
        self.genres = []
        self.year = None

    def process(self):
        if self.title[0] == '"' and self.title[-1]=='"':
            self.title = self.title[1:-1]
        if self.title[-1] == ')' and self.title[-7:-5] == ' (':
            self.year = int(self.title[-5:-1])
            self.title = self.title[:-7]

def load_movies(path, ignore_tags=False):
    # load movie basic info.
    movies = {}
    for line in file(path + '/movies.csv').readlines():
        # line may contains more then 2 ','
        row = line.strip().split(',')
        movie_id = row[0]
        title = ','.join(row[1:-1])
        genres = row[-1]
        
        if movie_id == 'movieId':
            # ignore first line.
            continue
    
        movie = MovieInfo()
        movie.id = int(movie_id)
        movie.title = title
        movie.genres = genres.split('|')
        movie.process()
        movies[movie.id] = movie
    pydev.info('load movie basic info over.')

    if ignore_tags:
        return movies

    # load tag meta-info.
    tag_info = {}
    for tagid, tag in pydev.foreach_row(file(path + '/genome-tags.csv'), seperator=','):
        if tagid == 'tagId':
            continue
        tag_info[tagid] = tag.strip()
    pydev.info('load tags info over.')

    # load genome tags info.
    tag_match_count = 0
    for movieid, tagid, score in pydev.foreach_row(file(path + '/genome-scores.csv'), seperator=','):
        try:
            key = int(movieid)
            if key not in movies:
                continue
            movies[key].tags.append( (int(tagid), tag_info.get(tagid, ''), float(score)) )
            tag_match_count += 1
        except Exception, e:
            pydev.err(e)

    # sort tags.
    pydev.info('sort tags..')
    for movie in movies:
        movies[movie].tags = sorted(movies[movie].tags, key=lambda x:-x[2])

    pydev.info('tag matchs : %d' % tag_match_count)

    return movies
    


def readfile(fd, test_num=-1):
    data = []
    for line in fd.readlines():
        uid, iid, score = line.split(',')
        uid = int(uid)
        iid = int(iid)
        score = int(score)
        data.append( (uid, iid, score))
        if test_num>0 and len(data)>=test_num:
            break
    return data

def readdata(dir, test_num=-1):
    # return data:
    #   train/valid: [(uid, iid, score), ..]
    #   test: [(uid, iid), (uid, iid), ..]

    print >> sys.stderr, 'load [%s/]' % dir
    train = readfile(file(dir + '/train'), test_num)
    valid = readfile(file(dir + '/valid'))
    test = readfile(file(dir + '/test'))
    
    print >> sys.stderr, 'load over'
    return train, valid, test

def measure(predictor, test, debug=False):
    progress = tqdm.tqdm(test)
    y = []
    y_ = []

    debug_fd = None
    if debug:
        debug_fd = file('log/debug.log', 'w')
    for uid, iid, score in progress:
        pred_score = predictor(uid, iid, debug_fd)
        if debug:
            print >> debug_fd, '%s\t%s\t%d\t%.3f' % (uid, iid, score, pred_score)
        
        y.append( score )
        y_.append( pred_score )

    pydev.info('Predict over')

    auc = metrics.roc_auc_score(y, y_)
    pydev.log('Test AUC: %.3f' % auc)
    
class Utils(pydev.InteractiveApp):
    def __init__(self):
        pydev.InteractiveApp.__init__(self)

    def load_movie(self, arg):
        path = 'data/ml-20m'
        if len(arg)>0:
            path = arg
        print path
        self.movies = load_movies(path)

    def movie(self, arg):
        mid = int(arg)
        movie = self.movies.get(mid, None)

        print 'Movie %d:' % (movie.id)
        print '[[ %s | %s ]]' % (movie.title, movie.year)
        print movie.genres
        for tid, tag, score in movie.tags[:20]:
            print '%s:%.3f' % (tag, score)

    def seek(self, arg):
        pattern = arg
        for movie in self.movies.values():
            if pattern in movie.title.lower():
                print '%s: %s' % (movie.id, movie.title)

    def load(self):
        train, valid, test = readdata(sys.argv[1])
        print len(train)
        print len(valid)
        print len(test)

    def help(self):
        print 'load_movie [<path>]: load movie info into memory.'
        print 'movie <movie_id> : seek movie by id.'
        print 'seek <query> : seek movie by pattern.'

if __name__=='__main__':
    ut = Utils()
    ut.run()
