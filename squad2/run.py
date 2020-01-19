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

if __name__=='__main__':
    data_path = '../dataset/squad2/'
    train_filename = data_path + 'train-v2.0.json'
    test_filename = data_path + 'dev-v2.0.json'

    #ider = nlp_utils.TokenID()
    #ider.add('<end>')

    vocab = torchtext.vocab.GloVe(name='6B')

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)

    #train_ques_tids, train_cont_tids, train_output, train_answer_range = squad_reader.load_data(train_reader, ider)
    train_ques_embs, train_cont_embs, train_output, train_answer_range = squad_reader.load_data_embeddings(train_reader, vocab)
    print >> sys.stderr, 'load train over'
    #test_ques_tids, test_cont_tids, test_output, test_answer_range = squad_reader.load_data(test_reader, ider)
    train_ques_embs, train_cont_embs, train_output, train_answer_range = squad_reader.load_data_embeddings(train_reader, vocab)
    print >> sys.stderr, 'load test over'
    #print >> sys.stderr, 'load data over (vocab=%d)' % (ider.size())

    sys.exit(0)

    # hyper-param.
    epoch_count=400
    batch_size = 128
    input_emb_size = 16
    hidden_size = 32

    # make model.
    #model = models.V0_Encoder(ider.size(), input_emb_size, hidden_size)
    model = models.V1_CatLstm(ider.size(), input_emb_size, hidden_size)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 10., 10.]).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger = file('log.txt', 'w')

    # training phase.
    for epoch in range(epoch_count):
        print 'Epoch %d' % epoch
        loss = 0 
        step = 0
        bar = tqdm.tqdm(range(0, len(train_ques_tids), batch_size))
        for s in bar:
            optimizer.zero_grad()

            batch_question_ids = rnn_utils.pad_sequence(train_ques_tids[s:s+batch_size], batch_first=True)
            batch_context_ids = rnn_utils.pad_sequence(train_cont_tids[s:s+batch_size], batch_first=True)

            temp_output = rnn_utils.pad_sequence(train_output[s:s+batch_size], batch_first=True)
            batch_context_output = torch.tensor(temp_output).cuda()

            y = model(batch_question_ids, batch_context_ids)

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
            test(model, train_ques_tids, train_cont_tids, train_output, train_answer_range, logger)
            test(model, test_ques_tids, test_cont_tids, test_output, test_answer_range, logger)






