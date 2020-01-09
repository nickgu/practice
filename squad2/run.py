#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import squad_reader
import torchtext 

class TokenID:
    def __init__(self):
        self.__idx2word = []
        self.__word2idx = {}

    def add(self, term):
        if term not in self.__word2idx:
            self.__word2idx[term] = len(self.__idx2word)
            self.__idx2word.append(term)

    def get_id(self, term):
        return self.__word2idx.get(term, -1)

    def get_term(self, id):
        if id < 0 or id >= len(self.__idx2word):
            return None
        return self.__idx2word[id]

    def size(self):
        return len(self.__idx2word)

    def __iter__(self):
        for term in self.__idx2word:
            yield term

if __name__=='__main__':
    data_path = '../dataset/squad2/'
    train_filename = data_path + 'train-v2.0.json'
    test_filename = data_path + 'dev-v2.0.json'

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    ider = TokenID()

    train_reader = squad_reader.SquadReader(train_filename)
    for title, doc in train_reader.iter_doc():
        for token in tokenizer(title):
            ider.add(token)
        for token in tokenizer(doc):
            ider.add(token)

    test_reader = squad_reader.SquadReader(test_filename)
    for title, doc in test_reader.iter_doc():
        for token in tokenizer(title):
            ider.add(token)
        for token in tokenizer(doc):
            ider.add(token)

    print ider.size()






