# -*- coding: utf-8 -*-
# gusimiu@baidu.com
# 

import sys
from collections import defaultdict

import cPickle
import numpy as np
import pydev

class Graph:
    def __init__(self, path):
        ts = pydev.TempStorage(sign=path, filename='graph.ts')
        if ts.has_data():
            self.__edges = ts.read()
            self.__users = ts.read()
            self.__items = ts.read()
            self.__user_sim_dict = ts.read()
            self.__item_sim_dict = ts.read()

        else:
            self.__edges = {}
            self.__users = defaultdict(list)
            self.__items = defaultdict(list)
            self.__user_sim_dict = defaultdict(dict)
            self.__item_sim_dict = defaultdict(dict)

            with open(path + '/ua.base') as fp:
                for line in fp.readlines():
                    user, item, rating, time = line.strip().split('\t')
                    rating = float(rating)

                    self.__edges[ (user, item) ] = rating
                    self.__users[ user ].append( (item, rating) )
                    self.__items[ item ].append( (user, rating) )

            print >> sys.stderr, 'calc user sim'
            for u1 in self.__users.keys():
                for u2 in self.__users.keys():
                    if u1 in self.__user_sim_dict[u2]:
                        continue
                    
                    sim = self.__calc_user_sim(u1, u2)
                    self.__user_sim_dict[u1][u2] = sim
                    self.__user_sim_dict[u2][u1] = sim
            print >> sys.stderr, 'calc user sim, over'
            
            print >> sys.stderr, 'calc item sim'
            for u1 in self.__items.keys():
                for u2 in self.__items.keys():
                    if u1 == u2:
                        continue
                    if u1 in self.__item_sim_dict[u2]:
                        continue
                    
                    sim = self.__calc_item_sim(u1, u2)
                    self.__item_sim_dict[u1][u2] = sim
                    self.__item_sim_dict[u2][u1] = sim
            print >> sys.stderr, 'calc item sim over.'

            print >> sys.stderr, 'Write data to ts.'
            ts.write(self.__edges)
            ts.write(self.__users)
            ts.write(self.__items)
            ts.write(self.__user_sim_dict)
            ts.write(self.__item_sim_dict)
            print >> sys.stderr, 'Write data to ts over.'

    def debug(self, user, item, users_info, items_info):
        user_ratings = map(lambda x:x[1], self.__users.get( user, [] ))
        item_ratings = map(lambda x:x[1], self.__items.get( item, [] ))
        print 'user rating: avg=%.2f, var=%.2f, count=%d' % ( 
                np.average(user_ratings),
                np.var(user_ratings),
                len(user_ratings) )

        print 'movie rating: avg=%.2f, var=%.2f, count=%d' % ( 
                np.average(item_ratings),
                np.var(item_ratings),
                len(item_ratings) )

        print 'user rating in sim items:'
        max_output_count = 10
        c = 0
        for s_item, s_sim in self.item_best_sim(item, count=1000):
            if (user, s_item) in self.__edges:
                print '%s sim=%.3f rate=%s\t%s' % (s_item, s_sim, self.__edges[user, s_item], items_info.debug(s_item, brief=True))
                c += 1
                if c >= max_output_count: break

        print 'items rating by sim user:'
        c = 0
        for s_user, s_sim in self.user_best_sim(user, count=1000):
            if (s_user, item) in self.__edges:
                print '%s sim=%.3f rate=%s\t%s' % (s_user, s_sim, self.__edges[s_user, item], users_info.debug(s_user, brief=True))
                c += 1
                if c >= max_output_count: break

    def __calc_user_sim(self, u1, u2):
        v1 = self.__users[u1]
        v2 = self.__users[u2]
        return self.__sim(v1, v2)

    def __calc_item_sim(self, i1, i2):
        v1 = self.__items[i1]
        v2 = self.__items[i2]
        return self.__sim(v1, v2)

    def __user_vectors(self, u):
        return self.__users[u]

    def __sim(self, v1, v2):
        factor = 1.5
        d = dict(v2)
        sim = 0
        tot = len(v1) + len(v2)
        for idx, value in v1:
            if idx in d:
                tot -= 1
                # 2**(5 - (diff[0, 4])) / 32. - 1.
                s = factor**(5 - abs(value - d[idx])) - 1.  
                s /= factor ** 5 -1.
                sim += s
        return sim * 1. / tot

    def user_best_sim(self, user, count=10):
        return sorted(self.__user_sim_dict[user].iteritems(), key=lambda x:-x[1])[:count]

    def item_best_sim(self, item, count=10):
        return sorted(self.__item_sim_dict[item].iteritems(), key=lambda x:-x[1])[:count]


class MovieManager:
    def __init__(self, path):
        self.__movie_info = {}
        for line in file(path + '/u.item').readlines():
            (id, title, date, vdate, imdb, unknown, action, adventure, animation, children, comedy, crime, documentary, drama, fantacy,
             noir, horror, musical, mystery, romance, scifi, thriller, war, western) = line.strip('\n').split('|')

            self.__movie_info[id] = {
                    'title' : title,
                    'imdb'  : imdb,
                    }
            if unknown == '1': self.__movie_info[id]['unknown'] = 1
            if action == '1': self.__movie_info[id]['action'] = 1
            if adventure == '1': self.__movie_info[id]['adventure'] = 1
            if animation == '1': self.__movie_info[id]['animation'] = 1
            if children == '1': self.__movie_info[id]['children'] = 1
            if comedy == '1': self.__movie_info[id]['comedy'] = 1
            if crime == '1': self.__movie_info[id]['crime'] = 1
            if documentary == '1': self.__movie_info[id]['documentary'] = 1
            if drama == '1': self.__movie_info[id]['drama'] = 1
            if fantacy == '1': self.__movie_info[id]['fantacy'] = 1
            if noir == '1': self.__movie_info[id]['noir'] = 1
            if horror == '1': self.__movie_info[id]['horror'] = 1
            if musical == '1': self.__movie_info[id]['musical'] = 1
            if mystery == '1': self.__movie_info[id]['mystery'] = 1
            if romance == '1': self.__movie_info[id]['romance'] = 1
            if scifi == '1': self.__movie_info[id]['scifi'] = 1
            if thriller == '1': self.__movie_info[id]['thriller'] = 1
            if war == '1': self.__movie_info[id]['war'] = 1
            if western == '1': self.__movie_info[id]['western'] = 1

    def debug(self, movie, brief=False):
        udata = self.__movie_info.get(movie, {})
        if brief:
            return 'movie: %s { %s }' % (movie, ', '.join(map(lambda x:'%s:%s'%(x[0][:4], x[1]), udata.iteritems())))
        else:
            return 'movie: %s {\n%s\n}' % (movie, ',\n'.join(map(lambda x:'  %s:%s'%(x[0], x[1]), udata.iteritems())))


class UserManager:
    def __init__(self, path):
        self.__user_info = {}
        with open(path + '/u.user') as fp:
            for line in fp.readlines():
                id, age, gender, occupation, zip = line.strip().split('|')
                self.__user_info[id] = {
                            'gender' : gender,
                            'occupation' : occupation,
                            'age' : age,
                            'zip' : zip,
                        }

    def debug(self, user, brief=False):
        udata = self.__user_info.get(user, {})
        if brief:
            return 'user: %s { %s }' % (user, ', '.join(map(lambda x:'%s:%s'%(x[0][:4], x[1]), udata.iteritems())))
        else:
            return 'user: %s {\n%s\n}' % (user, ',\n'.join(map(lambda x:'  %s:%s'%(x[0], x[1]), udata.iteritems())))


if __name__ == '__main__':
    path = sys.argv[1]
    print 'Waiting for load data ...'
    graph = Graph(path)
    movies = MovieManager(path)
    users = UserManager(path) 

    while 1:
        sys.stderr.write( '>>> ' )
        line = sys.stdin.readline()
        user, item = line.strip().split(' ')
        print 'user=%s, item=%s' % (user, item) 
        print users.debug(user)
        print movies.debug(item)
        graph.debug(user, item, users, movies)
