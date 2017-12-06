#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import os
import sys
import urllib
import numpy

class StringIndexer:
    def __init__(self):
        self.__dict = {}
        self.__cur = 0
    def seek(self, s):
        if s not in self.__dict:
            self.__dict[s] = self.__cur
            self.__cur += 1
        return self.__dict[s]

class DataSet(object):
    def __init__(self, data_name, download_urls, root_path = './dataset'):
        self._data_name = data_name
        self._download_urls = download_urls
        self._root_path = root_path
        self._data_path = self._root_path + '/' + self._data_name

    def data(self):
        if not os.path.exists(self._data_path):
            print >> sys.stderr, 'Data not exists, download from to [%s]'
            os.makedirs(self._data_path)
            self.download_data()

        return self._read_data()

    def download_data(self):
        for url, filename in self._download_urls:
            print >> sys.stderr, 'download data : [%s] from [%s]' % (filename, url)
            urllib.urlretrieve(url, self._data_path + '/' + filename)


class IrisData(DataSet):
    def __init__(self, root_path = './dataset'):
        DataSet.__init__(self, 'iris', download_urls = [(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data', 'iris.data'
            )], root_path=root_path)
    
    def _read_data(self):
        X = []
        Y = []
        for url, fn in self._download_urls:
            pathname = self._data_path + '/' + fn
            seeker = StringIndexer()
            for line in file(pathname).readlines():
                arr = line.strip().split(',')
                if len(arr)!=5:
                    continue
                x = numpy.array( map(lambda x:float(x), arr[:4]) )
                yidx = seeker.seek(arr[4])
                y = numpy.zeros(3)
                y[yidx] = 1.0
            
                X.append(x)
                Y.append(y)

        return numpy.array(X), numpy.array(Y)

if __name__=='__main__':
    pass
