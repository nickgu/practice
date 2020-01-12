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

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        self.__vocab_size = vocab_size
        self.__emb_size = emb_size
        self.__hidden_size = hidden_size

        self.__emb = torch.nn.Embedding(vocab_size, emb_size)
        self.__question_rnn = torch.nn.LSTM(emb_size, hidden_size, num_layers=3)
        self.__context_rnn = torch.nn.LSTM(emb_size, hidden_size, num_layers=3)
        self.__fc = torch.nn.Dense(3, activation='relu')

    def forward(self, question_tokens, context_tokens):
        q_emb = self.__emb(question_tokens)
        c_emb = self.__emb(context_tokens)
            
        _, q_hidden = self.__question_rnn(q_emb, hidden)
        out, _ = self.__context_rnn(c_emb, q_hidden)
        out = self.fc(out)

if __name__=='__main__':
    data_path = '../dataset/squad2/'
    train_filename = data_path + 'train-v2.0.json'
    test_filename = data_path + 'dev-v2.0.json'

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    ider = TokenID()

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)
    
    token2id(train_reader, test_reader, ider, tokenizer)
    print ider.size()

    # make model.
    emb = torch.nn.Embedding(iter.size())


    




