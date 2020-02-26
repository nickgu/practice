#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import json
import unicodedata
import string
import torch
import nlp_utils

def errlog(s):
    print(s, file=sys.stderr)

class SquadData:
    def __init__(self, data_name='SQuAD data'):
        self.data_name = data_name
        self.qtoks = []
        self.ctoks = []
        self.qchar = []
        self.cchar = []
        self.triple_output = []
        self.binary_output = []
        self.answer_range = []
        self.answer_candidates = []

    def shuffle(self):
        import random
        if len(self.qchar)==0:
            shuf = zip(self.qtoks, self.ctoks, self.triple_output, self.binary_output, self.answer_range, self.answer_candidates)
            random.shuffle(shuf)
            self.qtoks, self.ctoks, self.triple_output, self.binary_output, self.answer_range, self.answer_candidates = zip(*shuf)
        else:
            shuf = zip(self.qtoks, self.ctoks, self.triple_output, self.binary_output, self.answer_range, self.answer_candidates, self.qchar, self.cchar)
            random.shuffle(shuf)
            self.qtoks, self.ctoks, self.triple_output, self.binary_output, self.answer_range, self.answer_candidates, self.qchar, self.cchar = zip(*shuf)

class SquadReader():
    def __init__(self, filename):
        self.__data = json.loads(open(filename).read())

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
                    yield title.strip(), context, qid, question.strip(), ans, is_impossible

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


def padtoken(token):
    L=16
    token = (token + u'\0'*L)[:L]
    ret = map(lambda c:min(69999, ord(c)), token)
    return ret

def load_data(reader, tokenizer, data_name=None, limit_count=None, read_char=False):
    squad_data = SquadData(data_name)

    for title, context, qid, question, ans, is_impossible in reader.iter_instance():
        ans_start = -1
        ans_text = 'none'

        # ignore impossible first.
        # SQuAD 1.1 doesn't have impossible data.
        if is_impossible:
            continue

        if not is_impossible:
            ans_start = ans[0]['answer_start']
            ans_text = ans[0]['text']

        ans_cand = []
        for a in ans:
            ans_cand.append(a['text'])

        context_tokens = []
        answer_token_begin = -1
        answer_token_end = -1

        if context[ans_start:ans_start+len(ans_text)] == ans_text:
            a = context[:ans_start].strip()
            b = context[ans_start : ans_start + len(ans_text)].strip()
            c = context[ans_start+len(ans_text):].strip()

            context_tokens += tokenizer(a)
            answer_token_begin = len(context_tokens)
            context_tokens += tokenizer(b)
            # end is the ending position. not+1
            answer_token_end = len(context_tokens)
            context_tokens += tokenizer(c)
        else:
            context_tokens = tokenizer(context)
            errlog('Mismatch on answer finding..')

        context_tokens.append('<end>') # add a extra token.
        question_tokens = tokenizer(question)

        qt = []
        ct = []
        qchar = []
        cchar = []
        c_out = []
        b_out = []
        for tok in question_tokens:
            qt.append(tok)
            if read_char:
                qchar.append(padtoken(tok))
        for idx, tok in enumerate(context_tokens):
            ct.append(tok)
            if read_char:
                cchar.append(padtoken(tok))
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
        if read_char:
            squad_data.qchar.append(torch.tensor(qchar))
            squad_data.cchar.append(torch.tensor(cchar))
        squad_data.triple_output.append(torch.tensor(c_out))
        squad_data.binary_output.append(torch.tensor(b_out))
        squad_data.answer_range.append( (answer_token_begin, answer_token_end) )
        squad_data.answer_candidates.append( ans_cand )
        
        if limit_count is not None:
            limit_count -= 1
            if limit_count <=0:
                break

    return squad_data



