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
import pydev

def run_test(model, test, batch_size, logger=None):
    # test code.
    with torch.no_grad():
        count = 0
        exact_match = 0
        side_match = 0
        
        loss = 0 
        step = 0

        for s in tqdm.tqdm(range(0, len(test.qtoks), batch_size)):
            batch_qt = test.qtoks[s:s+batch_size]
            batch_ques_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).cuda()
            batch_ct = test.ctoks[s:s+batch_size]
            batch_cont_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).cuda()

            y = model(batch_ques_emb, batch_cont_emb)
            # triple.
            p = y.softmax(dim=2)
            # for softmax on pos.
            ans = p.permute((0,2,1)).max(dim=2).indices
            # accumulate loss.
            temp_output = rnn_utils.pad_sequence(test.triple_output[s:s+batch_size], batch_first=True)
            batch_context_output = torch.tensor(temp_output).cuda()
            y = y.view(-1, 3)
            l = criterion(y.view(-1,3), batch_context_output.view([-1]))
           
            step += 1
            loss += l

            # for softmax on pos.
            for idx, ((a,b), (c,d)) in enumerate(zip(ans[:, 1:].tolist(), test.answer_range[s:s+batch_size])):
                # test one side.
                count += 1
                if a==c and b==d:
                    exact_match += 1

                if a==c:
                    side_match += 1
                if b==d:
                    side_match += 1
        info = 'EM=%.2f%% (%d/%d), one_side=%.2f%% Loss=%.5f' % (
                exact_match*100./count, exact_match, count, 
                side_match*50./count, loss/step)

        print info
        if logger:
            print >> logger, info

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
    arg = pydev.Arg('SQuAD data training program with pytorch.')
    arg.str_opt('epoch', 'e', default='200')
    arg.str_opt('batch', 'b', default='64')
    arg.str_opt('test_epoch', 't', default='5')
    arg.str_opt('logname', 'L', default='log.txt')
    opt = arg.init_arg()

    logger = file(opt.logname, 'w')

    # hyper-param.
    epoch_count = int(opt.epoch)
    batch_size = int(opt.batch)
    test_epoch = int(opt.test_epoch)

    input_emb_size = 400

    print >> sys.stderr, 'epoch=%d' % epoch_count
    print >> sys.stderr, 'test_epoch=%d' % test_epoch
    print >> sys.stderr, 'batch_size=%d' % batch_size
    print >> sys.stderr, 'log_filename=%s' % opt.logname


    # === Init Model ===
    #
    # make model.
    #model = models.V0_Encoder(ider.size(), input_emb_size, hidden_size)
    #model = models.V1_CatLstm(input_emb_size, hidden_size, layer_num=layer_num, dropout=0.4)
    model = models.V2_MatchAttention(input_emb_size).cuda()
    #model = models.V2_1_BiDafLike(input_emb_size).cuda()
    #model = models.V3_CrossConv().cuda()
    #model = models.V4_Transformer(input_emb_size).cuda()

    print ' == model_size: ', easy_train.model_params_size(model), ' =='
    # for triple.
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 100., 100.]).cuda())
    # for binary.
    #criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 100.]).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # === Init Data ===
    data_path = '../dataset/squad1/'
    train_filename = data_path + 'train-v1.1.json'
    test_filename = data_path + 'dev-v1.1.json'

    print >> sys.stderr, 'train: [%s]' % train_filename
    print >> sys.stderr, 'test: [%s]' % test_filename

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 

    #unk_emb = UnkEmb()
    #vocab = torchtext.vocab.GloVe(name='6B', unk_init=unk_emb.get)
    #vocab = torchtext.vocab.GloVe(name='6B', unk_init=lambda t:torch.randn(300))
    #vocab = torchtext.vocab.GloVe(name='6B')
    vocab = nlp_utils.TokenEmbeddings()

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)

    train = squad_reader.load_data(train_reader, tokenizer, limit_count=None)
    test = squad_reader.load_data(test_reader, tokenizer, limit_count=None)
    print >> sys.stderr, 'Load data over, train=%d, test=%d' % (len(train.qtoks), len(test.qtoks))

    # pre-heat.
    preheat(vocab, train.qtoks, train.ctoks, test.qtoks, test.ctoks)

    #check_coverage(train.ctoks, vocab)
    #check_coverage(test.ctoks, vocab)

    # training phase.
    for epoch in range(epoch_count):
        print 'Epoch %d' % epoch
        loss = 0 
        step = 0
        bar = tqdm.tqdm(range(0, len(train.qtoks), batch_size))
        for s in bar:
            optimizer.zero_grad()

            batch_qt = train.qtoks[s:s+batch_size]
            batch_ques_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).cuda()
            batch_ct = train.ctoks[s:s+batch_size]
            batch_cont_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).cuda()
            batch_start_end = torch.tensor(train.answer_range[s:s+batch_size]).cuda()

            # triple.
            temp_output = rnn_utils.pad_sequence(train.triple_output[s:s+batch_size], batch_first=True)
            batch_context_output = torch.tensor(temp_output).cuda()

            #y = model(batch_question_ids, batch_context_ids)
            y = model(batch_ques_emb, batch_cont_emb)
            p = y.softmax(dim=2)

            # softmax on each pos.
            # triple
            y = y.view(-1, 3)
            l = criterion(y.view(-1,3), batch_context_output.view([-1]))
            # binary.
            #y = y.view(-1, 2)
            #l = criterion(y.view(-1,2), batch_context_output.view([-1]))

            l.backward()
            #model.check_gradient()
            optimizer.step()

            step += 1
            loss += l
            bar.set_description('loss=%.5f' % (loss / step))

        print >> logger, 'epoch=%d\tloss=%.5f' % (epoch, loss/step)

        if test_epoch>0:
            if (epoch+1) % test_epoch ==0:
                print >> logger, 'Epoch %d:' % epoch

                run_test(model, train, batch_size, logger)
                run_test(model, test, batch_size, logger)

    run_test(model, train, batch_size, logger)
    run_test(model, test, batch_size, logger)

