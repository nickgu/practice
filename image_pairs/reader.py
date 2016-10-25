#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import struct
import cv2
import numpy as np

class DBReader:
    def __init__(self):
        pass

    def read(self, filename):
        self.__fd = file(filename)
        self.__record_count = 0
        
        while 1:
            ret = self.__read_buffers(3)

            # load over.
            if ret is None:
                break
            else:
                buf_label, buf_image_a, buf_image_b = ret
                #print len(buf_label), len(buf_image_a), len(buf_image_b)
                
                label = struct.unpack('i', buf_label)[0]
                image_a = np.np.asarray(bytearray(buf_image_a), dtype=np.uint8)
                image_b = np.np.asarray(bytearray(buf_image_b), dtype=np.uint8)

                yield label, image_a, image_b
                self.__record_count += 1

        pydev.log('%d image(s) loaded' % self.__record_count)
    
    def __read_buffers(self, count):
        ret = []
        for i in range(count):
            buf = self.__read_single_buffer()
            if buf is None:
                return None
            ret.append(buf)
        return ret

    def __read_single_buffer(self):
        # load 8 bytes length.
        m = self.__fd.read(8)
        if len(m)<8:
            return None
        length = struct.unpack('Q', m)[0]
        buff = self.__fd.read(length)
        if len(buff)<length:
            return None
        return buff

if __name__=='__main__':
    filename = sys.argv[1]

    labels = []
    Xs = []
    Ys = []
    for label, im_1, im_2 in DBReader().read(filename):
        labels.append(label)
        Xs.append(im_1)
        Ys.append(im_2)

    labels = numpy.array(labels)
    Xs = numpy.array(Xs)
    Ys = numpy.array(Ys)
