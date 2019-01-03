#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import pydev
import random

class UnitTest(pydev.App):
    def __init__(self):
        pydev.App.__init__(self)

    def slot_index(self):
        import easy_train
        coder = easy_train.SlotIndexCoder()
        for i in range(100):
            slot = 'slot_%s' % i
            for j in range(random.randint(30, 100)):
                key = 'key_%s' % j
        
                coder.alloc(slot, key)

        coder.save(file('test.idx', 'w'))
        
        second_coder = easy_train.SlotIndexCoder()
        second_coder.load(file('test.idx'))
        
        assert(coder == second_coder)


if __name__=='__main__':
    app = UnitTest()
    app.run()
