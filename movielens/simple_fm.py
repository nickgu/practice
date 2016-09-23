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

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm

class SimpleFMLearner:
    def __init__(self, iter=100):
        # Build and train a Factorization Machine
        self.__fm = pylibfm.FM(num_factors=10, num_iter=iter, 
                verbose=True, 
                task="regression", 
                initial_learning_rate=0.001, 
                learning_rate_schedule="optimal")

    def fit(self, train):
        ''' train : [(userid, itemid, rating)...] '''
        train_data = []
        y_train = []
        for userid, itemid, rating in train:
            train_data.append({ "user_id": str(userid), "movie_id": str(itemid)})
            y_train.append( rating )

        self.__v = DictVectorizer()
        X_train = self.__v.fit_transform(train_data)
        self.__fm.fit(X_train,y_train)

    def predict(self, userid, itemid):
        X_test = self.__v.transform(
                [
                    { "user_id": str(userid), "movie_id": str(itemid)}
                ]
                )
        preds = self.__fm.predict(X_test)
        return preds[0]

# Read in data
def loadData(filename,path="/Users/nickgu/lab/datasets/movielens/ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

class FM:
    def __init__(a):
        pass

if __name__ == '__main__':
    (train_data, y_train, train_users, train_items) = loadData("ua.base")
    (test_data, y_test, test_users, test_items) = loadData("ua.test")
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




