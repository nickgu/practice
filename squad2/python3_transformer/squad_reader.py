#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import json
import unicodedata
import string
import torch
import py3dev
import nlp_utils

from transformers import *

class SquadData:
    def __init__(self, data_name='SQuAD data'):
        self.data_name = data_name
        self.qtoks = []
        self.ctoks = []
        self.triple_output = []
        self.binary_output = []
        self.answer_range = []
        self.answer_candidates = []

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


def load_data(reader, tokenizer, data_name=None, limit_count=None):
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
            py3dev.error('Mismatch on answer finding..')

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
        squad_data.answer_candidates.append( ans_cand )
        
        if limit_count is not None:
            limit_count -= 1
            if limit_count <=0:
                break

    return squad_data

def check_answer(answer_fn, squad_fn, output_fn):
    import sys
    import tqdm

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    answer_fd = open(answer_fn)
    answers = {}
    for row in py3dev.foreach_row(open(answer_fn)):
        if len(row)!=7:
            break
        pred, tgt, tag, pratio, py_, py, ori_index = row
        pratio = float(pratio)
        py_ = float(py_)
        py = float(py)
        ps, pe = map(lambda x:int(x), pred.split(','))
        ts, te = map(lambda x:int(x), tgt.split(','))
        answers[int(ori_index)] = ((ps, pe), (ts, te), tag, pratio, py_, py)
    print('answer loaded.', file=sys.stderr)

    reader = SquadReader(squad_fn)
    output = open(output_fn, 'w')
    idx = -1
    count = 0
    n_EM = 0
    n_SM = 0
    n_SM_B = 0
    n_SM_E = 0
    print('answers=%d' % len(answers))
    bar = tqdm.tqdm(reader.iter_instance())
    for title, context, qid, question, ans, is_impossible in bar:
        idx += 1
        if is_impossible:
            continue
        if idx not in answers:
            continue
        
        qtoks = tokenizer.tokenize(question)
        offset = len(qtoks) + 2
        ctoks = tokenizer.tokenize(context)
        ans_y_, ans_y, tag, p_ratio, p_y_, p_y = answers[idx]

        ans_y_ = list(map(lambda x:x-offset, ans_y_))
        ans_y = list(map(lambda x:x-offset, ans_y))

        print('\n## ID=%d ##\n%s' % (idx, '='*100), file=output)
        print('== Context ==', file=output)
        print(context, file=output)
        print('== Context Tokens ==', file=output)
        print((u','.join(ctoks)), file=output)
        print('== Question ==', file=output)
        print(question, file=output)
        print('== Question Tokens ==', file=output)
        print((u','.join(qtoks)), file=output)
        print('== Expected answer ==', file=output)
        correct_answer = tokenizer.convert_tokens_to_string(ctoks[ans_y[0]:ans_y[1]])
        print('rec: ' + correct_answer, file=output)
        print('(%d, %d)' % (ans_y[0], ans_y[1]), file=output)
        for a in ans:
            print('%s (%d)' % (a['text'], a['answer_start']), file=output)
        print('== Predict output ==', file=output)

        ori_answer = tokenizer.convert_tokens_to_string(ctoks[ans_y_[0]:ans_y_[1]])
        print(ori_answer, file=output)
        print('(%d, %d)' % (ans_y_[0], ans_y_[1]), file=output)

        # match candidate or both side match.
        em = False
        if ans_y_ == ans_y:
            em = True

        adjust_answer = ori_answer.replace(' ', '')
        for a in ans:
            aa = a['text'].lower().replace(u' ', u'')
            if aa == adjust_answer:
                em = True
                break

        if em:
            print(' ## ExactMatch!', file=output)
            n_EM += 1
        elif ans_y_[0] == ans_y[0] or ans_y_[1]==ans_y[1]:
            print((' ## SideMatch! [%s]' % tag), file=output)
            n_SM += 1
            if tag == 'SM_B': n_SM_B += 1
            if tag == 'SM_E': n_SM_E += 1
        else:
            print(' ## Wrong!', file=output)
        print('p_ratio=%.3f, p_y_=%.5f, p_y=%.5f' % (p_ratio, p_y_, p_y), file=output)

        count += 1
        bar.set_description('EM=%.1f%%(%d), SM=%.1f%%, B=%.1f%%, E=%.1f%%, N=%d' % (
            n_EM * 100. / count, n_EM,
            n_SM * 100. / count,
            n_SM_B * 100. / count, n_SM_E * 100. / count,
            count
            ))

def check_ans_train():
    check_answer('log/ans/train.ans.out', 
            '../../../practice/dataset/squad1/train-v1.1.json', 
            'log/ans/check_ans.train.out')

def check_ans_test():
    check_answer('log/ans/test.ans.out', 
            '../../../practice/dataset/squad1/dev-v1.1.json', 
            'log/ans/check_ans.test.out')

if __name__=='__main__':
    import fire
    fire.Fire()

