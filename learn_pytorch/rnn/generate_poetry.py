#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 
# this program learn and generate 
# chinese peotry by gru.
#
# dataset from: 
#   https://github.com/nickgu/chinese-poetry
#

import os
import json
import sys
import random

import gru
import torch


class PoetryRepo:
    def __init__(self, path='dataset/poetry'):
        use_poem = False # use poem or sentence.
        self.__all_poetry = []
        test_count = 50
        output_file = file('poet.debug.txt', 'w')
        for filename in os.listdir(path):
            if filename.startswith('poet.'):
                with file(path + '/' + filename) as fd:
                    d = json.loads(fd.read())
                    for item in d:
                        if 'paragraphs' in item:
                            poet = u''.join(item['paragraphs'])
                            # whole poem.
                            if use_poem:
                                self.__all_poetry.append( poet ) 
                                print >> output_file, poet.encode('utf8')
                            else:
                                # single sentence.
                                for s in poet.split(u'，'):
                                    for u in s.split(u'。'):
                                        if len(u)>0:
                                            #print u.encode('utf-8')
                                            self.__all_poetry.append( u ) 
                                            print >> output_file, u.encode('utf8')

                            if test_count > 0 and len(self.__all_poetry)>=test_count:
                                break
                if test_count > 0 and len(self.__all_poetry)>=test_count:
                    break

        
        print >> sys.stderr, 'load over [%d poets loaded.]' % len(self.__all_poetry)
        print >> sys.stderr, '%s' % self.__all_poetry[random.randint(0, len(self.__all_poetry)-1)].encode('utf-8')


    def __iter__(self):
        for item in self.__all_poetry:
            yield item

if __name__=='__main__':
    lang = gru.Lang('poetry')
    poetry_repo = PoetryRepo()

    datas = []
    poets = []

    for poet in poetry_repo:
        vec = lang.addSentence(poet)
        poets.append(poet)
        datas.append(vec)

    print >> sys.stderr, 'data_counts=%d' % len(datas)
    print >> sys.stderr, 'word_counts=%d' % lang.n_words

    epoch_count = 2000
    hidden_size = 32

    encoder = gru.EncoderRNN(lang.n_words, hidden_size)

    decoder = gru.AttnDecoderRNN(hidden_size, lang.n_words, dropout_p=0.1)
    trainFunc = gru.trainIters
    evaluationFunc = gru.evaluate

    #decoder = gru.DecoderRNN(lang.n_words, hidden_size)
    #trainFunc = gru.trainItersRnn
    #evaluationFunc = gru.evaluateRnn

    if False:
        encoder.cuda()
        decoder.cuda()

    argset = set()
    if len(sys.argv)>1:
        argset = set(sys.argv[1:])

    if '--load' in argset:
        print >> sys.stderr, 'load model'
        encoder.load_state_dict(torch.load('encoder.pkl'))
        decoder.load_state_dict(torch.load('decoder.pkl'))

    if '--train' in argset:
        print >> sys.stderr, 'training'
        training_pairs = [(poet[:-1], poet[1:]) for poet in datas]

        print >> sys.stderr, 'traning samples: %s' % (str(training_pairs[0][0]))
        for i in range(epoch_count):
            print >> sys.stderr, 'epoch: %d' % i
            trainFunc(training_pairs, encoder, decoder)

            idx = random.randint(0, len(datas)-1)
            input = poets[idx][:-1]
            output = evaluationFunc(input, encoder, decoder, lang, lang)
            print '>>> ' + u''.join(input).encode('utf-8')
            print '>>> ' + ','.join(map(lambda x:'%d'%x, datas[idx][:-1]))
            print '<<< ' + u''.join(output).encode('utf-8')

            ri = random.randint(0, len(poets)-1)
            input = poets[ri][:-1]
            output, output_hidden = gru.evaluate(input, encoder, decoder, lang, lang)
            print >> sys.stderr, '>>> ' + u''.join(input).encode('utf8')
            print >> sys.stderr, '>>> ' + ','.join(map(lambda x:'%d'%x, datas[ri][:-1]))
            print >> sys.stderr, '--- ' + u''.join(poets[ri][1:]).encode('utf8')
            print >> sys.stderr, '<<< ' + u''.join(output).encode('utf8')


        print >> sys.stderr, 'train over.'
        torch.save(encoder.state_dict(), 'encoder.pkl')
        torch.save(decoder.state_dict(), 'decoder.pkl')
        print >> sys.stderr, 'save model.'

    input = poets[0][:-1]
    output = evaluationFunc(input, encoder, decoder, lang, lang)
    print '>>> ' + u''.join(input).encode('utf-8')
    print '>>> ' + ','.join(map(lambda x:'%d'%x, datas[0][:-1]))
    print '<<< ' + u''.join(output).encode('utf-8')


    print >> sys.stderr, 'read char to write poet.'
    input_line = sys.stdin.readline().decode('utf-8').strip()
    print >> sys.stderr, 'you input [%s] to generate.' % (input_line.encode('utf8'))

    gen = ['SOS', first_char]
    for i in range(20):
        output = evaluationFunc(gen, encoder, decoder, lang, lang)
        print u''.join(output).encode('utf-8')
        gen.append(output[-1])

