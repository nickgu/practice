#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import json
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"

class SquadData:
    def __init__(self):
        self.qtoks = []
        self.ctoks = []
        self.triple_output = []
        self.binary_output = []
        self.answer_range = []

class SquadReader():
    def __init__(self, filename):
        self.__data = json.loads(file(filename).read())

    def iter_instance(self):
        '''
            return (title, paragraph, qid, question, ans, is_impossible)
        '''
        for item in self.__data['data']:
            title = item['title']
            for para in item['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    qid = qa['id']
                    question = qa['question']
                    ans = qa['answers']
                    is_impossible = qa.get('is_impossible', False)
                    yield title, context, qid, question, ans, is_impossible

    def iter_doc(self):
        for item in self.__data['data']:
            title = item['title']
            doc = u''
            for para in item['paragraphs']:
                context = para['context']
                doc += context + '\n'
            yield title, doc

    def iter_question(self):
        for item in self.__data['data']:
            for para in item['paragraphs']:
                for qa in para['qas']:
                    qid = qa['id']
                    question = qa['question']
                    ans = qa['answers']
                    is_impossible = qa.get('is_impossible', False)
                    yield qid, question, ans, is_impossible

def load_data(reader, tokenizer, limit_count=None):
    import torch
    squad_data = SquadData()

    for title, context, qid, question, ans, is_impossible in reader.iter_instance():
        ans_start = -1
        ans_text = 'none'

        # ignore impossible first.
        if is_impossible:
            continue

        if not is_impossible:
            ans_start = ans[0]['answer_start']
            ans_text = ans[0]['text']

        context_tokens = []
        answer_token_begin = -1
        answer_token_end = -1

        if context[ans_start:ans_start+len(ans_text)] == ans_text:
            a = context[:ans_start]
            b = context[ans_start : ans_start + len(ans_text)]
            c = context[ans_start+len(ans_text):]

            context_tokens += tokenizer(a)
            answer_token_begin = len(context_tokens)
            context_tokens += tokenizer(b)
            answer_token_end = len(context_tokens)
            context_tokens += tokenizer(c)
        else:
            context_tokens = tokenizer(context)

        context_tokens.append('<end>')
        question_tokens = tokenizer(question)

        #all_context_tokens.append( context_tokens )
        #all_question_tokens.append( question_tokens )

        # question_tokens, context_tokens, context_output(0,1,2)
        qt = []
        ct = []
        c_out = []
        b_out = []
        for tok in question_tokens:
            qt.append(tok)
        for idx, tok in enumerate(context_tokens):
            ct.append(tok)
            if idx == answer_token_begin:
                c_out.append(1)
                b_out.append((1,0))
            elif idx == answer_token_end:
                c_out.append(2)
                b_out.append((0,1))
            else:
                c_out.append(0)
                b_out.append((0,0))

        squad_data.qtoks.append(qt)
        squad_data.ctoks.append(ct)
        squad_data.triple_output.append(torch.tensor(c_out))
        squad_data.binary_output.append(torch.tensor(b_out))
        squad_data.answer_range.append( (answer_token_begin, answer_token_end) )
        
        if limit_count is not None:
            limit_count -= 1
            if limit_count <=0:
                break

    return squad_data

def count_tokens(reader):
    import torchtext 

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    maxl = 0
    for title, context, qid, question, ans, is_impossible in reader.iter_instance():
        l = len(list(tokenizer(title))) + len(list(tokenizer(context))) + len(list(tokenizer(question)))
        if l > maxl:
            maxl = l

    print 'maxl:', maxl

def list_titles(reader):
    for title, doc in reader.iter_doc():
        print title.encode('gb18030') + ' (%s)' % len(doc)


if __name__=='__main__':
    import sys
    path = sys.argv[1]
    reader = SquadReader(path)

    #count_tokens(reader)
    list_titles(reader)





