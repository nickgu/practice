#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import json
import unicodedata
import string
import fire
import pydev

class SquadData:
    def __init__(self):
        self.qtoks = []
        self.ctoks = []
        self.triple_output = []
        self.binary_output = []
        self.answer_range = []

    def shuffle(self):
        import random
        shuf = zip(self.qtoks, self.ctoks, self.triple_output, self.binary_output, self.output, self.answer_range)
        random.shuffle(shuf)
        self.qtoks, self.ctoks, self.triple_output, self.binary_output, self.output, self.answer_range = zip(*shuf)

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

def load_data(reader, tokenizer, limit_count=None):
    import torch
    squad_data = SquadData()

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
            answer_token_end = len(context_tokens)
            context_tokens += tokenizer(c)
        else:
            context_tokens = tokenizer(context)
            print >> sys.stderr, 'Mismatch on answer finding..'

        #context_tokens.append('<end>')
        question_tokens = tokenizer(question)

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

def debug_length(squad_fn):
    reader = SquadReader(squad_fn)
    maxl = 0
    maxs = 0
    maxn = 0
    maxq = None

    import torchtext 
    tk = torchtext.data.utils.get_tokenizer('revtok') # case sensitive.
    tokenizer = lambda s: map(lambda u:u.strip(), tk(s))

    for title, context, qid, question, ans, is_impossible in reader.iter_instance():
        l = len(question)
        s = len(question.split(' '))
        n = len(tokenizer(question))
        if l > maxl:
            maxl = l
            maxq = question
        if s > maxs:
            maxs = s
        if n > maxn:
            maxn = n

    print maxq
    print 'maxl:', maxl
    print 'maxs:', maxs
    print 'maxn:', maxn


def list_titles(reader):
    for title, doc in reader.iter_doc():
        print title.encode('gb18030') + ' (%s)' % len(doc)

def check_conflict(reader):
    stat_dict = {}
    all = 0
    for title, context, qid, question, ans, is_impossible in reader.iter_instance():
        if is_impossible:
            continue
        c = context.count(ans[0]['text'])
        all += 1
        stat_dict[c] = stat_dict.get(c, 0) + 1

    for n, x  in sorted(stat_dict.iteritems(), key=lambda x:-x[1]):
        print n, x, '%.2f%%' % (x*100/all)

def check_answer(answer_fn, squad_fn, output_fn):
    import torchtext 
    import sys
    #tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    tk = torchtext.data.utils.get_tokenizer('revtok') # case sensitive.
    tokenizer = lambda s: map(lambda u:u.strip(), tk(s))

    answer_fd = file(answer_fn)
    answers = []
    for row in pydev.foreach_row(file(answer_fn)):
        if len(row)!=3:
            break
        pred, tgt, total = row
        ps, pe = map(lambda x:int(x), pred.split(','))
        ts, te = map(lambda x:int(x), tgt.split(','))
        answers.append(((ps, pe), (ts, te)))
    print >> sys.stderr, 'answer loaded.'

    reader = SquadReader(squad_fn)
    output = file(output_fn, 'w')
    idx = 0
    for title, context, qid, question, ans, is_impossible in reader.iter_instance():
        if is_impossible:
            continue
        if idx >= len(answers):
            break
        
        qtoks = tokenizer(question)
        ctoks = tokenizer(context)
        ans_info = answers[idx]

        print >> output, '\n## ID=%d ##\n%s' % (idx, '='*100)
        print >> output, '== Context =='
        print >> output, context.encode('utf8')
        print >> output, '== Context Tokens =='
        print >> output, (u','.join(ctoks)).encode('utf8')
        print >> output, '== Question =='
        print >> output, question.encode('utf8')
        print >> output, '== Question Tokens =='
        print >> output, (u','.join(qtoks)).encode('utf8')
        print >> output, '== Expected answer =='
        print >> output, 'rec: ' + u' '.join(ctoks[ans_info[1][0]:ans_info[1][1]]).encode('utf8')
        print >> output, '(%d, %d)' % (ans_info[1][0], ans_info[1][1])
        for a in ans:
            print >> output, '%s (%d)' % (a['text'].encode('utf8'), a['answer_start'])
        print >> output, '== Predict output =='
        print >> output, u' '.join(ctoks[ans_info[0][0]:ans_info[0][1]]).encode('utf8')
        print >> output, '(%d, %d)' % (ans_info[0][0], ans_info[0][1])
        if ans_info[0] == ans_info[1]:
            print >> output, ' ## ExactMatch!'
        elif ans_info[0][0] == ans_info[1][0] or ans_info[0][1]==ans_info[1][1]:
            print >> output, ' ## SideMatch!'
        else:
            print >> output, ' ## Wrong!'

        idx += 1
        

if __name__=='__main__':
    fire.Fire()


