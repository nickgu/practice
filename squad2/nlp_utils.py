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


class TokenEmbeddings:
    def __init__(self):
        self.__cache = {}
        self.__vocab = torchtext.vocab.GloVe(name='6B')
        self.__char_emb = torchtext.vocab.CharNGram()

    def preheat(self, tokens):
        temp_toks = []
        for tok in tokens:
            if tok in self.__cache:
                continue
            temp_toks.append(tok)
        
        if len(temp_toks)>0:
            embs = self.get_vecs_by_tokens_inner(temp_toks)
            for idx, tok in enumerate(temp_toks):
                self.__cache[tok] = embs[idx]

    def cache_size(self):
        return len(self.__cache)

    def get_vecs_by_tokens(self, tokens):
        l = []
        for tok in tokens:
            l.append(self.__cache[tok])
        return torch.stack(l)

    def get_vecs_by_tokens_inner(self, tokens):
        # dim=300
        word_emb = self.__vocab.get_vecs_by_tokens(tokens)
        # dim=100
        char_emb = self.__char_emb.get_vecs_by_tokens(tokens).view(-1, 100)
        return torch.cat((word_emb, char_emb), dim=1)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def u2a(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def rand_init():
    return torch.randn(300) * 1e-3

if __name__=='__main__':
    pass
