#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import json
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def u2a(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


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
                    is_impossible = qa['is_impossible']
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
                    is_impossible = qa['is_impossible']
                    yield qid, question, ans, is_impossible

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





