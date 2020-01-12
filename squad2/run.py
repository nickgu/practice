#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import squad_reader
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
        return out

if __name__=='__main__':
    data_path = '../dataset/squad2/'
    train_filename = data_path + 'train-v2.0.json'
    test_filename = data_path + 'dev-v2.0.json'

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    ider = TokenID()

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)
    
    '''
    token2id(train_reader, test_reader, ider, tokenizer)
    print ider.size()
    '''

    count = 0
    train_question = []
    train_context = []
    train_output = []
    for title, context, qid, question, ans, is_impossible in train_reader.iter_instance():
        ans_start = -1
        ans_text = 'none'
        if not is_impossible:
            ans_start = ans[0]['answer_start']
            ans_text = ans[0]['text']

        context_tokens = []
        answer_token_first = -1
        answer_token_last = -1

        if context[ans_start:ans_start+len(ans_text)] == ans_text:
            a = context[:ans_start]
            b = context[ans_start : ans_start + len(ans_text)]
            c = context[ans_start+len(ans_text):]

            context_tokens += tokenizer(a)
            answer_token_first = len(context_tokens)
            context_tokens += tokenizer(b)
            answer_token_last = len(context_tokens)-1
            context_tokens += tokenizer(c)
        else:
            context_tokens = tokenizer(context)

        question_tokens = tokenizer(question)

        # question_tokens, context_tokens, context_output(0,1,2)
        q_ids = []
        c_ids = []
        c_out = []
        for tok in question_tokens:
            q_ids.append( ider.add(tok) )
        for idx, tok in enumerate(context_tokens):
            c_ids.append( ider.add(tok) )
            if idx == answer_token_first:
                c_out.append(1)
            elif idx == answer_token_last:
                c_out.append(2)
            else:
                c_out.append(0)

        train_question.append(q_ids)
        train_context.append(c_ids)
        train_output.append(c_out)

    print >> file('temp/q', 'w'), train_question[:10]
    print >> file('temp/c', 'w'), train_context[:10]
    print >> file('temp/o', 'w'), train_output[:10]


    # make model.
    #model = Encoder()




    




