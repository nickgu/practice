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

    def read(self):
        for item in self.__data['data']:
            title = u2a(item['title'])
            for para in item['paragraphs']:
                context = u2a(para['context'])
                for qa in para['qas']:
                    qid = qa['id']
                    question = u2a(qa['question'])
                    ans = qa['answers']
                    is_possbible = qa['is_impossible']
                    yield title, context, qid, question, ans, is_possbible

if __name__=='__main__':
    path = '../dataset/squad2/train-v2.0.json'
    reader = SquadReader(path)
    for title, context, qid, question, ans, is_possbible in reader.read():
        print question
