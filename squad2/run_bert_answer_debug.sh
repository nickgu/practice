#! /bin/sh

python squad_reader.py check_answer \
    python3_transformer/log/ans/train.ans.out \
    ../../practice/dataset/squad1/train-v1.1.json \
    python3_transformer/log/ans/check_ans.train.out

python squad_reader.py check_answer \
    python3_transformer/log/ans/test.ans.out \
    ../../practice/dataset/squad1/dev-v1.1.json \
    python3_transformer/log/ans/check_ans.test.out
