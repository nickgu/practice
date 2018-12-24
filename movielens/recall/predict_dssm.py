#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import random

import torch
import torch.nn as nn

import pydev
import utils
import embedding_dict


def algor_dssm(train, valid, test, topN):
    index = embedding_dict.EmbeddingDict('temp/dssm_out_emb.txt', contain_key=False, metric='angular')
    embeddingBag = nn.EmbeddingBag(131263, 64, mode='mean')
    embeddingBag.state_dict = torch.load('temp/dssm.pkl')
    
    def predict(uid, items):
        readset = set(map(lambda x:x[0], items))

        # same code for input.
        InputSize = 10
        input_nids = []
        input_offset = []

        input = []
        for j in range(InputSize):
            input.append( items[random.randint(0, len(items)-1)][0] )

        input_offset.append(len(input_nids))
        input_nids += input

        print input_offset
        print input_nids

        inputs_emb = embeddingBag(
                torch.tensor(input_nids), 
                torch.tensor(input_offset)
                )
        input_emb = inputs_emb[0]

        print input_emb
                
        ans, dis = index.index.get_nns_by_vector(input_emb, n=100, include_distances=True)
        ret = []
        for item, score in zip(ans, dis):
            if item in readset:
                continue

            ret.append(str(item))
            if len(ret)>=TopN:
                return ret

    utils.measure(predict, test, debug=True)



if __name__=='__main__':
    TopN = 10
    TestNum = 100

    print >> sys.stderr, 'begin loading data..'
    train, valid, test = utils.readdata('data', test_num=TestNum)
    print >> sys.stderr, 'load over'

    print >> sys.stderr, 'Algor: DSSM'
    algor_dssm(train, valid, test, TopN)

