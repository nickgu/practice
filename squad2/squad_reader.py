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
import torch
import nlp_utils

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
            print >> sys.stderr, 'Mismatch on answer finding..'

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

def debug_length(squad_fn):
    reader = SquadReader(squad_fn)
    maxl = 0
    maxs = 0
    maxn = 0
    maxq = None

    tokenizer = nlp_utils.init_tokenizer()
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

def answer_length_distribution(squad_fn):
    import torchtext 
    import sys
    import tqdm

    tokenizer = nlp_utils.init_tokenizer()
    reader = SquadReader(squad_fn)
    dist = {}
    total = 0
    for title, context, qid, question, ans, is_impossible in tqdm.tqdm(reader.iter_instance()):
        if is_impossible:
            continue
        for a in ans:
            toks = tokenizer(a['text'])
            l = len(toks)
            dist[l] = dist.get(l, 0) + 1
            total += 1
    
    acc_perc = 0
    for length, count in sorted(dist.iteritems(), key=lambda x:x[0])[:30]:
        perc = count*100./total
        acc_perc += perc
        print 'length=%d\t%d\t%.1f%%\t%.1f%%' % (
                length, count, perc, acc_perc)
    
def check_answer(answer_fn, squad_fn, output_fn):
    import torchtext 
    import sys
    import tqdm

    tokenizer = nlp_utils.init_tokenizer()
    answer_fd = file(answer_fn)
    answers = []
    for row in pydev.foreach_row(file(answer_fn)):
        if len(row)!=6:
            break
        pred, tgt, tag, pratio, py_, py = row
        pratio = float(pratio)
        py_ = float(py_)
        py = float(py)
        ps, pe = map(lambda x:int(x), pred.split(','))
        ts, te = map(lambda x:int(x), tgt.split(','))
        answers.append(((ps, pe), (ts, te), tag, pratio, py_, py))
    print >> sys.stderr, 'answer loaded.'

    reader = SquadReader(squad_fn)
    output = file(output_fn, 'w')
    idx = 0
    n_EM = 0
    n_SM = 0
    n_SM_B = 0
    n_SM_E = 0
    bar = tqdm.tqdm(reader.iter_instance())
    for title, context, qid, question, ans, is_impossible in bar:
        if is_impossible:
            continue
        if idx >= len(answers):
            break
        
        qtoks = tokenizer(question)
        ctoks = tokenizer(context)
        ans_y_, ans_y, tag, p_ratio, p_y_, p_y = answers[idx]

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
        print >> output, 'rec: ' + u' '.join(ctoks[ans_y[0]:ans_y[1]]).encode('utf8')
        print >> output, '(%d, %d)' % (ans_y[0], ans_y[1])
        for a in ans:
            print >> output, '%s (%d)' % (a['text'].encode('utf8'), a['answer_start'])
        print >> output, '== Predict output =='
        print >> output, u' '.join(ctoks[ans_y_[0]:ans_y_[1]]).encode('utf8')
        print >> output, '(%d, %d)' % (ans_y_[0], ans_y_[1])

        # match candidate or both side match.
        em = False
        if ans_y_ == ans_y:
            em = True
        adjust_answer = u''.join(ctoks[ans_y_[0]:ans_y_[1]]).replace(u' ', u'')
        for a in ans:
            aa = a['text'].replace(u' ', u'')
            if aa == adjust_answer:
                em = True
                break

        if em:
            print >> output, ' ## ExactMatch!'
            n_EM += 1
        elif ans_y_[0] == ans_y[0] or ans_y_[1]==ans_y[1]:
            print >> output, (' ## SideMatch! [%s]' % tag)
            n_SM += 1
            if tag == 'SM_B': n_SM_B += 1
            if tag == 'SM_E': n_SM_E += 1
        else:
            print >> output, ' ## Wrong!'
        print >> output, 'p_ratio=%.3f, p_y_=%.5f, p_y=%.5f' % (p_ratio, p_y_, p_y)

        idx += 1
        bar.set_description('EM=%.1f%%(%d), SM=%.1f%%, B=%.1f%%, E=%.1f%%, N=%d' % (
            n_EM * 100. / idx, n_EM,
            n_SM * 100. / idx,
            n_SM_B * 100. / idx, n_SM_E * 100. / idx,
            idx
            ))
        

if __name__=='__main__':
    fire.Fire()


