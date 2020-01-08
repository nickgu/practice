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
                    is_impossbible = qa['is_impossible']
                    yield title, context, qid, question, ans, is_impossbible

if __name__=='__main__':
    import sys
    path = sys.argv[1]
    reader = SquadReader(path)
    answer_dict = {}
    for title, context, qid, question, ans, is_impossible in reader.read():
        answer_dict[qid] = ''
        if is_impossible:
            print question, is_impossible

    print >> file('ans.out', 'w'),  json.dumps(answer_dict)
