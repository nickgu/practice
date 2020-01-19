#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
import torchtext

class TokenID:
    def __init__(self):
        self.__idx2word = []
        self.__word2idx = {}

    def add(self, term):
        if term not in self.__word2idx:
            self.__word2idx[term] = len(self.__idx2word)
            self.__idx2word.append(term)
        return self.__word2idx[term]

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

def token2id(train_reader, test_reader, ider, tokenizer):
    for title, doc in train_reader.iter_doc():
        for token in tokenizer(title):
            ider.add(token)
        for token in tokenizer(doc):
            ider.add(token)

    for qid, q, _, _ in train_reader.iter_question():
        for token in tokenizer(q):
            ider.add(token)

    for title, doc in test_reader.iter_doc():
        for token in tokenizer(title):
            ider.add(token)
        for token in tokenizer(doc):
            ider.add(token)

    for qid, q, _, _ in test_reader.iter_question():
        for token in tokenizer(q):
            ider.add(token)

def rand_init():
    return torch.randn(300) * 1e-3

if __name__=='__main__':
    pass
