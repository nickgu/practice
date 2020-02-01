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

sys.path.append('../learn_pytorch')
import easy_train


def test(model, ques_tids, cont_tids, output, answer_range, batch_size, logger=None):
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


def test_toks(model, ques_toks, cont_toks, output, answer_range, batch_size, logger=None):
    # test code.
    with torch.no_grad():
        count = 0
        correct = 0
        loss = 0 
        step = 0

        for s in range(0, len(ques_toks), batch_size):
            batch_qt = ques_toks[s:s+batch_size]
            batch_ques_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).cuda()
            batch_ct = cont_toks[s:s+batch_size]
            batch_cont_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).cuda()

            y = model(batch_ques_emb, batch_cont_emb)
            p = y.softmax(dim=2)
            
            # for softmax on pos.
            ans = p.permute((0,2,1)).max(dim=2).indices
            
            # for softmax on seq.
            #ans = p.max(dim=2).indices

            # accumulate loss.
            temp_output = rnn_utils.pad_sequence(output[s:s+batch_size], batch_first=True)
            batch_context_output = torch.tensor(temp_output).cuda()
            y = y.view(-1, 3)
            l = criterion(y.view(-1,3), batch_context_output.view([-1]))
            step += 1
            loss += l

            # for softmax on pos.
            for idx, ((a,b), (c,d)) in enumerate(zip(ans[:, 1:].tolist(), answer_range[s:s+batch_size])):
            # for softmax on seq.
            #for idx, ((a,b), (c,d)) in enumerate(zip(ans.tolist(), answer_range[s:s+batch_size])):
                count += 2
                if a==c:
                    correct += 1
                if b==d:
                    correct += 1

        print 'Precise=%.2f%% (%d/%d), Loss=%.5f' % (correct*100./count, correct, count, loss/step)
        if logger:
            print >> logger, 'Precise=%.2f%% (%d/%d), Loss=%.5f' % (correct*100./count, correct, count, loss/step)

def check_coverage(toks, vocab):
    count = 0
    hit = 0
    for sentence in toks:
        count += len(sentence)
        m = vocab.get_vecs_by_tokens(sentence)
        hit += len(filter(lambda x:x, [i.sum().abs()>1e-5 for i in m]))

    print 'Vocab coverage: %.2f%% (%d/%d)' % (hit*100./count, hit, count)

class UnkEmb:
    def __init__(self):
        self.__dct={}
    def get(self, tok):
        if tok not in self.__dct:
            emb = torch.randn(300) * 1e-3
            self.__dct[tok] = emb
            return emb
        return self.__dct[tok]

def preheat(vocab, *args):
    for doc in args:
        for sentence in doc:
            vocab.preheat(sentence)
    print 'Pre-heat over', vocab.cache_size()
    
if __name__=='__main__':
    data_path = '../dataset/squad1/'
    train_filename = data_path + 'train-v1.1.json'
    test_filename = data_path + 'dev-v1.1.json'

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    #ider = nlp_utils.TokenID()
    #ider.add('<end>')

    #unk_emb = UnkEmb()
    #vocab = torchtext.vocab.GloVe(name='6B', unk_init=unk_emb.get)
    #vocab = torchtext.vocab.GloVe(name='6B', unk_init=lambda t:torch.randn(300))
    #vocab = torchtext.vocab.GloVe(name='6B')
    vocab = nlp_utils.TokenEmbeddings()

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)

    #train_ques_tids, train_cont_tids, train_output, train_answer_range = squad_reader.load_data(train_reader, tokenizer, ider)
    train_ques_toks, train_cont_toks, train_output, train_answer_range = squad_reader.load_data_tokens(train_reader, tokenizer, limit_count=None)
    print >> sys.stderr, 'load train over'
    #test_ques_tids, test_cont_tids, test_output, test_answer_range = squad_reader.load_data(test_reader, tokenizer, ider)
    test_ques_toks, test_cont_toks, test_output, test_answer_range = squad_reader.load_data_tokens(test_reader, tokenizer, limit_count=None)
    print >> sys.stderr, 'load test over'
    #print >> sys.stderr, 'load data over (vocab=%d)' % (ider.size())

    # pre-heat.
    preheat(vocab, train_ques_toks, train_cont_toks, test_ques_toks, test_cont_toks)

    #check_coverage(train_ques_toks, vocab)
    #check_coverage(test_ques_toks, vocab)

    # hyper-param.
    epoch_count=400
    batch_size = 64
    input_emb_size = 400
    hidden_size = 256
    layer_num = 2

    # make model.
    #model = models.V0_Encoder(ider.size(), input_emb_size, hidden_size)
    #model = models.V1_CatLstm(input_emb_size, hidden_size, layer_num=layer_num, dropout=0.4)
    model = models.V2_MatchAttention(input_emb_size, hidden_size, layer_num=layer_num, dropout=0.4)

    print ' == model_size: ', easy_train.model_params_size(model), ' =='

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 100., 100.]).cuda())
    #criterion = torch.nn.CrossEntropyLoss()
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
            batch_start_end = torch.tensor(train_answer_range[s:s+batch_size]).cuda()

            #batch_question_ids = rnn_utils.pad_sequence(train_ques_tids[s:s+batch_size], batch_first=True)
            #batch_context_ids = rnn_utils.pad_sequence(train_cont_tids[s:s+batch_size], batch_first=True)

            temp_output = rnn_utils.pad_sequence(train_output[s:s+batch_size], batch_first=True)
            batch_context_output = torch.tensor(temp_output).cuda()

            #y = model(batch_question_ids, batch_context_ids)
            y = model(batch_ques_emb, batch_cont_emb)
            p = y.softmax(dim=2)

            # softmax on each pos.
            y = y.view(-1, 3)
            l = criterion(y.view(-1,3), batch_context_output.view([-1]))

            # softmax on whole seq.
            #print p.shape
            #print batch_start_end.shape
            #l = criterion(p.view(len(batch_qt)*2, -1), batch_start_end.view([-1]))
            
            l.backward()
            #model.check_gradient()
            optimizer.step()

            step += 1
            loss += l
            bar.set_description('loss=%.5f' % (loss / step))
            #sys.exit(0)

        if (epoch+1) % 3 ==0:
            print >> logger, 'Epoch %d:' % epoch
            #test(model, train_ques_tids, train_cont_tids, train_output, train_answer_range, logger)
            #test(model, test_ques_tids, test_cont_tids, test_output, test_answer_range, logger)

            test_toks(model, train_ques_toks, train_cont_toks, train_output, train_answer_range, batch_size, logger)
            test_toks(model, test_ques_toks, test_cont_toks, test_output, test_answer_range, batch_size, logger)





