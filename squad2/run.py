#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import squad_reader
import torch
import torchtext 
import torch.nn.utils.rnn as rnn_utils
import sys
import tqdm
import models
import nlp_utils


def test(model, ques_tids, cont_tids, output, answer_range, logger=None):
    # test code.
    batch_size = 128
    with torch.no_grad():
        count = 0
        correct = 0

        for s in range(0, len(ques_tids), batch_size):
            batch_question_ids = rnn_utils.pad_sequence(ques_tids[s:s+batch_size], batch_first=True)
            batch_context_ids = rnn_utils.pad_sequence(cont_tids[s:s+batch_size], batch_first=True)

            y = model(batch_question_ids, batch_context_ids)
            p = y.softmax(dim=2)
            ans = p.permute((0,2,1)).max(dim=2).indices

            for idx, ((a,b), (c,d)) in enumerate(zip(ans[:, 1:].tolist(), answer_range[s:s+batch_size])):
                count += 2
                if a==c:
                    correct += 1
                if b==d:
                    correct += 1

        print 'Precise=%.2f%% (%d/%d)' % (correct*100./count, correct, count)
        if logger:
            print >> logger,'Precise=%.2f%% (%d/%d)' % (correct*100./count, correct, count)

def test_toks(model, ques_toks, cont_toks, output, answer_range, logger=None):
    # test code.
    batch_size = 128
    with torch.no_grad():
        count = 0
        correct = 0

        for s in range(0, len(ques_toks), batch_size):
            batch_qt = ques_toks[s:s+batch_size]
            batch_ques_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).cuda()
            batch_ct = cont_toks[s:s+batch_size]
            batch_cont_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).cuda()

            y = model(batch_ques_emb, batch_cont_emb)
            p = y.softmax(dim=2)
            ans = p.permute((0,2,1)).max(dim=2).indices

            for idx, ((a,b), (c,d)) in enumerate(zip(ans[:, 1:].tolist(), answer_range[s:s+batch_size])):
                count += 2
                if a==c:
                    correct += 1
                if b==d:
                    correct += 1

        print 'Precise=%.2f%% (%d/%d)' % (correct*100./count, correct, count)
        if logger:
            print >> logger,'Precise=%.2f%% (%d/%d)' % (correct*100./count, correct, count)

if __name__=='__main__':
    data_path = '../dataset/squad2/'
    train_filename = data_path + 'train-v2.0.json'
    test_filename = data_path + 'dev-v2.0.json'

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    #ider = nlp_utils.TokenID()
    #ider.add('<end>')
    vocab = torchtext.vocab.GloVe(name='6B')

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)

    #train_ques_tids, train_cont_tids, train_output, train_answer_range = squad_reader.load_data(train_reader, tokenizer, ider)
    train_ques_toks, train_cont_toks, train_output, train_answer_range = squad_reader.load_data_tokens(train_reader, tokenizer)
    print >> sys.stderr, 'load train over'
    #test_ques_tids, test_cont_tids, test_output, test_answer_range = squad_reader.load_data(test_reader, tokenizer, ider)
    test_ques_toks, test_cont_toks, test_output, test_answer_range = squad_reader.load_data_tokens(test_reader, tokenizer)
    print >> sys.stderr, 'load test over'
    #print >> sys.stderr, 'load data over (vocab=%d)' % (ider.size())

    # hyper-param.
    epoch_count=400
    batch_size = 128
    input_emb_size = 300
    hidden_size = 64
    layer_num = 3

    # make model.
    #model = models.V0_Encoder(ider.size(), input_emb_size, hidden_size)
    model = models.V1_CatLstm(input_emb_size, hidden_size, layer_num=layer_num)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 10., 10.]).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger = file('log.txt', 'w')

    # training phase.
    for epoch in range(epoch_count):
        print 'Epoch %d' % epoch
        loss = 0 
        step = 0
        bar = tqdm.tqdm(range(0, len(train_ques_toks), batch_size))
        for s in bar:
            optimizer.zero_grad()

            batch_qt = train_ques_toks[s:s+batch_size]
            batch_ques_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).cuda()
            batch_ct = train_cont_toks[s:s+batch_size]
            batch_cont_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).cuda()

            #batch_question_ids = rnn_utils.pad_sequence(train_ques_tids[s:s+batch_size], batch_first=True)
            #batch_context_ids = rnn_utils.pad_sequence(train_cont_tids[s:s+batch_size], batch_first=True)

            temp_output = rnn_utils.pad_sequence(train_output[s:s+batch_size], batch_first=True)
            batch_context_output = torch.tensor(temp_output).cuda()

            #y = model(batch_question_ids, batch_context_ids)
            y = model(batch_ques_emb, batch_cont_emb)

            p = y.softmax(dim=2)
            y = y.view(-1, 3)
            l = criterion(y.view(-1,3), batch_context_output.view([-1]))
            l.backward()
            optimizer.step()

            step += 1
            loss += l
            bar.set_description('loss=%.5f' % (loss / step))

        if (epoch+1) % 5 ==0:
            print >> logger, 'Epoch %d:' % epoch
            #test(model, train_ques_tids, train_cont_tids, train_output, train_answer_range, logger)
            #test(model, test_ques_tids, test_cont_tids, test_output, test_answer_range, logger)

            test_toks(model, train_ques_toks, train_cont_toks, train_output, train_answer_range, logger)
            test_toks(model, test_ques_toks, test_cont_toks, test_output, test_answer_range, logger)





