#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 
#   NOTICE: how to install pyfm:
#   Run this cmd:
# 
#   sudo pip install Cython
#   sudo pip install git+https://github.com/coreylynch/pyFM
#
#

import sys
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tffm
import config

class Info: 
    def __init__(self, path='./'):
        self.__global_user_info = {}
        self.__global_movie_info = {} 
        self.__movie_info = {}
        self.__user_info = {}

        OutputMovieProperty     = True
        OutputUserProperty      = False
        OutputNeighborInfo      = True

        UserFeatureAsGlobal     = False
        UserFeatureAsUser       = True

        MovieFeatureAsGlobal    = False
        MovieFeatureAsMovie     = True

        NeighborRateThreshold   = 2
        #NeighborAsGlobal        = False

        # OUTPUT MOVIE PROPERTY
        if OutputMovieProperty:
            for line in file(path + config.ItemData).readlines():
                if config.DataType == config.D_100k:
                    (id, title, date, vdate, imdb, unknown, action, adventure, animation, children, comedy, crime, documentary, drama, fantacy,
                     noir, horror, musical, mystery, romance, scifi, thriller, war, western) = line.strip('\n').split(config.PropertySeperator)

                    feature_value = {
                            'unknown' : int(unknown),
                            'action': int(action),
                            'adventure': int(adventure),
                            'animation': int(animation),
                            'children': int(children),
                            'comedy': int(comedy),
                            'crime' : int(crime),
                            'documentary': int(documentary),
                            'drama' : int(drama),
                            'fantacy' : int(fantacy),
                            'noir' : int(noir),
                            'horror' : int(horror),
                            'musical' : int(musical),
                            'mystery' : int(mystery),
                            'romance' : int(romance),
                            'scifi' : int(scifi),
                            'thriller' : int(thriller),
                            'war' : int(war),
                            'western' : int(western)
                            }
                    
                else:
                    (id, title, types) = line.strip('\n').split(config.PropertySeperator)
                    feature_value = {}
                    for tag in types.split('|'):
                        feature_value[tag] = 1

                if MovieFeatureAsGlobal:
                    self.__global_movie_info[int(id)] = feature_value
                if MovieFeatureAsMovie:
                    self.__movie_info[int(id)] = feature_value

        # OUTPUT USER PROPERTY
        if OutputUserProperty:
            for line in file(path + config.UserData).readlines():
                id, age, gender, occupation, zip = line.strip().split(config.PropertySeperator)
                feature_value = {
                            'gender' : gender,
                            'occupation' : occupation,
                            'age' : str(int(age) / 5)
                        }
                if UserFeatureAsGlobal:
                    self.__global_user_info[int(id)] = feature_value
                if UserFeatureAsUser:
                    self.__user_info[int(id)] = feature_value


        # OUTPUT NEIGHBORHOOD PROPERTY
        # try to add neighborhood info. (significant improval for svdfeature)
        # use ua.base as oringinal data source.
        #   add user feature: movie_user_rated.
        #   add movie feature: user_rated.
        if OutputNeighborInfo:
            neighbor_count = 0
            for line in file(path + config.TrainData).readlines():
                uid, mid, rate, time = line.strip('\n').split(config.Seperator)
                uid = int(uid)
                mid = int(mid)
                rate = int(rate)
                if rate < NeighborRateThreshold:
                    continue

                neighbor_count += 1
                if uid not in self.__global_user_info:
                    self.__global_user_info[uid] = {}
                '''
                if mid not in self.__global_movie_info:
                    self.__global_movie_info[mid] = {}
                '''

                self.__global_user_info[uid]['movie_%s' % mid] = 1
                #self.__global_movie_info[mid]['user_%s' % uid] = 1

            print >> sys.stderr, 'NeighborCount = %d' % neighbor_count

    def process(self, userid, movieid, data):
        udata = self.__user_info.get(userid, {})
        for key, value in udata.iteritems():
            data[key] = value
        mdata = self.__movie_info.get(movieid, {})
        for key, value in mdata.iteritems():
            data[key] = value
        return data

    def process_global(self, userid, movieid, data):
        # try to patch user feature to global features.
        udata = self.__global_user_info.get(userid, {})
        for key, value in udata.iteritems():
            data[key] = value
        mdata = self.__global_movie_info.get(movieid, {})
        for key, value in mdata.iteritems():
            data[key] = value
        return data

    def process_user(self, userid, movieid, data):
        udata = self.__user_info.get(userid, {})
        for key, value in udata.iteritems():
            data[key] = value
        return data

    def process_movie(self, userid, movieid, data):
        mdata = self.__movie_info.get(movieid, {})
        for key, value in mdata.iteritems():
            data[key] = value
        return data


class SimpleFMLearner:
    def __init__(self, iter=100, factor=10, use_info=True, path='./', external_fm=None):
        from pyfm import pylibfm
        self.__use_info = use_info
        # temp code, load ml-100k's info
        if self.__use_info:
            self.__info = Info(path)

        # Build and train a Factorization Machine
        if external_fm:
            print >> sys.stderr, 'Use external FM: %s' % type(external_fm)
            self.__fm = external_fm
        else:
            print >> sys.stderr, 'iter=%d, factor=%d, use_info=%d' % (iter, factor, use_info)
            self.__fm = pylibfm.FM(num_factors=factor, num_iter=iter, 
                    verbose=True, 
                    task="regression", 
                    initial_learning_rate=0.001, 
                    learning_rate_schedule="optimal")



    def fit(self, train):
        ''' train : [(userid, itemid, rating)...] '''
        train_data = []
        y_train = []
        for userid, itemid, rating in train:
            d = self.__make_data(userid, itemid)

            train_data.append(d)
            y_train.append( rating )

        self.__v = DictVectorizer()


        X_train = self.__v.fit_transform(train_data)
        y_train = np.array(y_train)

        print >> sys.stderr, 'x_train.shape=%s, type=%s' % (str(X_train.shape), type(X_train))
        print >> sys.stderr, 'y_train.shape=%s, type=%s' % (str(y_train.shape), type(y_train))
        if isinstance(self.__fm, tffm.models.TFFMRegressor):
            self.__fm.fit(X_train, y_train, show_progress=True)
        else:
            self.__fm.fit(X_train, y_train)
        print >> sys.stderr, 'Train completed.'

    def predict(self, userid, itemid):
        d = self.__make_data(userid, itemid)
        X_test = self.__v.transform([d])
        preds = self.__fm.predict(X_test)
        return preds[0]

    def __make_data(self, userid, itemid):
        userid = int(userid)
        itemid = int(itemid)
        d = { "user_id": str(userid), "movie_id": str(itemid)}
        if self.__use_info:
            d = self.__info.process(userid, itemid, d)
        return d

# Read in data
def loadData(filename, info, path="./"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            d = { "user_id": str(user), "movie_id": str(movieid)}
            d = info.process(int(user), int(movieid), d)

            data.append(d)
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

if __name__ == '__main__':
    info = Info()
    
    (train_data, y_train, train_users, train_items) = loadData("ua.base", info)
    (test_data, y_test, test_users, test_items) = loadData("ua.test", info)
    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)

    print X_train.shape
    print y_train.shape

    # Build and train a Factorization Machine
    fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")

    fm.fit(X_train,y_train)

    # Evaluate
    preds = fm.predict(X_test)
    from sklearn.metrics import mean_squared_error
    print("FM MSE: %.4f" % mean_squared_error(y_test,preds))




