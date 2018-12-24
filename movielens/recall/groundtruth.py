#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev
import utils

def algor_hot(train, valid, test, topN):
    stat = {}
    for uid, items in train:
        for iid, score, ts in items:
            if iid not in stat:
                stat[iid] = [0, 0]

            stat[iid][score] += 1

    top = sorted(stat.iteritems(), key=lambda x:-x[1][1])
    print 'stat over'
    #print top

    def predict(uid, items):
        readset = set(map(lambda x:x[0], items))

        ans = []
        for item in map(lambda x:x[0], top):
            if item in readset:
                continue
            ans.append(item)
            if len(ans) == topN:
                break
        return ans[:topN]

    utils.measure(predict, test)

def algor_cooc(train, valid, test, topN, only1=False):
    # using dict built by build_cooc.py
    fd = file('temp/cooc.txt') 

    cooc_dict = {}
    for key, items in pydev.foreach_row(fd):
        items = map(lambda x:(x[0], int(x[1])), map(lambda x:x.split(':'), items.split(',')))
        cooc_dict[key] = items
    print >> sys.stderr, 'cooc load over'

    def predict(uid, items):
        local_stat = {}
        readset = set(map(lambda x:x[0], items))

        for item, score, _ in items:
            if only1 and score!=1:
                continue
            cooc_items = cooc_dict.get(item, [])
            for c_item, c_count in cooc_items:
                if c_item in readset:
                    continue
                local_stat[c_item] = local_stat.get(c_item, 0) + c_count

        ans = map(lambda x:x[0], sorted(local_stat.iteritems(), key=lambda x:-x[1])[:topN])
 
        '''
        print 'items:'
        print items
        print 'local:'
        print sorted(local_stat.iteritems(), key=lambda x:-x[1])[:20]
        print 'ans:'
        print ans
        '''

        return ans

    utils.measure(predict, test, debug=False)
        

def algor_item2vec(train, valid, test, topN):
    import embedding_dict
    index = embedding_dict.ItemIndex('temp/word2vec.output.txt', 500)
    
    def predict(uid, items):
        readset = set(map(lambda x:x[0], items))

        stat_dict = {}
        search_error = 0
        for idx, score, ts in items:
            if score != 1:
                continue
            idx = int(idx)
            try:
                ans, dis = index.index.get_nns_by_item(idx, n=300, include_distances=True)
                #print idx, ans
                #print dis
 
                for item, score in zip(ans, dis):
                    if item == idx:
                        continue
                    stat_dict[item] = stat_dict.get(item, 0) + score

            except:
                search_error += 1
                continue
               
        ans = sorted(stat_dict.iteritems(), key=lambda x:-x[1])
        ret = []
        for item, score in ans:
            if item in readset:
                continue
            ret.append(str(item))
            if len(ret)>=topN:
                return ret
        return ret

    utils.measure(predict, test, debug=False)

def algor_embeddings(train, valid, test, topN):
    import embedding_dict
    index = embedding_dict.EmbeddingDict('temp/nid_emb.txt', contain_key=False)
    
    def predict(uid, items):
        readset = set(map(lambda x:x[0], items))

        stat_dict = {}
        search_error = 0
        for idx, score, ts in items:
            if score != 1:
                continue
            idx = int(idx)
            try:
                ans, dis = index.index.get_nns_by_item(idx, n=50, include_distances=True)
                #print idx, ans
                #print dis
 
                for item, score in zip(ans, dis):
                    if item == idx:
                        continue
                    stat_dict[item] = stat_dict.get(item, 0) + score

            except:
                search_error += 1
                continue
               
        ans = sorted(stat_dict.iteritems(), key=lambda x:-x[1])
        ret = []
        for item, score in ans:
            if item in readset:
                continue
            ret.append(str(item))
            if len(ret)>=topN:
                return ret
        return ret

    utils.measure(predict, test, debug=False)



if __name__=='__main__':
    TopN = 10
    TestNum = 100

    print >> sys.stderr, 'begin loading data..'
    train, valid, test = utils.readdata('data', test_num=TestNum)
    print >> sys.stderr, 'load over'

    print >> sys.stderr, 'Algor: Embeddings'
    algor_embeddings(train, valid, test, TopN)

    '''
    print >> sys.stderr, 'Algor: Hot'
    algor_hot(train, valid, test, TopN)

    print >> sys.stderr, 'Algor: Cooc'
    algor_cooc(train, valid, test, TopN)

    print >> sys.stderr, 'Algor: CoocOnly_1'
    algor_cooc(train, valid, test, TopN, only1=True)
    '''

    '''
    print >> sys.stderr, 'Algor: Item2Vec'
    algor_item2vec(train, valid, test, TopN)
    '''


        
    
