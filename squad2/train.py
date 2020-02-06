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

# import models.
from models import *
from milestones import *

sys.path.append('../learn_pytorch')
import easy_train
import pydev
import nlp_utils

class RunTypeBinary:
    def __init__(self):
        self.__criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 100.]).cuda())

    def loss(self, predict, target):
        batch_context_output = rnn_utils.pad_sequence(target, batch_first=True).detach().cuda()
        l = self.__criterion(predict.view(-1,2), batch_context_output.view([-1]))
        return l

    def get_ans_range(self, y):
        return y.permute(0,2,1,3)[:,:,:,1:].squeeze().max(dim=2).indices

    def adjust_data(self, data):
        data.output = data.binary_output
          

class RunTypeTriple:
    def __init__(self):
        self.__criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 100., 100.]).cuda())

    def loss(self, predict, target):
        batch_context_output = rnn_utils.pad_sequence(target, batch_first=True).detach().cuda()
        l = self.__criterion(predict.view(-1,3), batch_context_output.view([-1]))
        return l

    def get_ans_range(self, y):
        return y.permute((0,2,1)).max(dim=2).indices[:,1:]

    def adjust_data(self, data):
        data.output = data.triple_output


def run_test(runtype, model, data, batch_size, logger=None, answer_output=None):
    # test code.
    with torch.no_grad():
        count = 0
        exact_match = 0
        side_match = 0
        
        loss = 0 
        step = 0

        for s in tqdm.tqdm(range(0, len(data.qtoks), batch_size)):
            batch_qt = data.qtoks[s:s+batch_size]
            batch_ques_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).cuda()
            batch_ct = data.ctoks[s:s+batch_size]
            batch_cont_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).cuda()

            batch, clen, _ = batch_cont_emb.shape
            y = rnn_utils.pad_sequence(data.output[s:s+batch_size], batch_first=True)

            # triple output: (batch, clen, 3)
            # binary output: (batch, clen, 2, 2)
            y_ = model(batch_ques_emb, batch_cont_emb)

            l = runtype.loss(y_, y)
            step += 1
            loss += l.item()

            ans = runtype.get_ans_range(y_)
            for idx, ((a,b), (c,d)) in enumerate(zip(ans.tolist(), data.answer_range[s:s+batch_size])):
                if answer_output:
                    print >> answer_output, '%d,%d\t%d,%d\t%d' % (a,b,c,d, len(data.ctoks[s+idx]))
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

        print >> sys.stderr, info
        if logger:
            print >> logger, info

def check_coverage(toks, vocab):
    count = 0
    hit = 0
    for sentence in toks:
        count += len(sentence)
        m = vocab.get_vecs_by_tokens(sentence)
        hit += len(filter(lambda x:x, [i.sum().abs()>1e-5 for i in m]))

    print >> sys.stderr, 'Vocab coverage: %.2f%% (%d/%d)' % (hit*100./count, hit, count)

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
    print >> sys.stderr, 'Pre-heat over', vocab.cache_size()
    
if __name__=='__main__':
    arg = pydev.Arg('SQuAD data training program with pytorch.')
    arg.str_opt('epoch', 'e', default='200')
    arg.str_opt('batch', 'b', default='64')
    arg.str_opt('test_epoch', 't', default='5')
    arg.str_opt('logname', 'L', default='log.txt')
    arg.str_opt('save', 's', default='params/temp_model.pkl')
    arg.bool_opt('continue_training', 'c')
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
    print >> sys.stderr, 'save_model=%s' % opt.save
    print >> sys.stderr, 'continue_training=%s' % opt.continue_training

    # === Init Model ===
    
    runtype = RunTypeTriple()
    #runtype = RunTypeBinary()

    # milestones model.
    #model = V2_MatchAttention(input_emb_size).cuda()

    # research model.
    model = V2_MatchAttention(input_emb_size, hidden_size=128, dropout=0.2).cuda()
    #model = V2_MatchAttention_Test(input_emb_size).cuda()
    #model = V2_MatchAttention_Binary(input_emb_size).cuda()
    #model = V0_Encoder(ider.size(), input_emb_size, hidden_size)
    #model = V1_CatLstm(input_emb_size, hidden_size, layer_num=layer_num, dropout=0.4)
    #model = V3_FCEmbModel(input_emb_size).cuda()
    #model = V2_2_BilinearAttention(input_emb_size).cuda()
    #model = V2_1_BiDafLike(input_emb_size).cuda()
    #model = V3_CrossConv().cuda()
    #model = V4_Transformer(input_emb_size).cuda()

    print >> sys.stderr, ' == model_size: ', easy_train.model_params_size(model), ' =='

    if opt.continue_training:
        print >> sys.stderr, 'prepare to load previous model.'
        model.load_state_dict(torch.load(opt.save))
        print >> sys.stderr, 'load over.'

    # for triple.
    # criterion init in runtype.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # === Init Data ===
    data_path = '../dataset/squad1/'
    train_filename = data_path + 'train-v1.1.json'
    test_filename = data_path + 'dev-v1.1.json'

    print >> sys.stderr, 'train: [%s]' % train_filename
    print >> sys.stderr, 'test: [%s]' % test_filename

    #tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    tk = torchtext.data.utils.get_tokenizer('revtok') # case sensitive.
    tokenizer = lambda s: map(lambda u:u.strip(), tk(s))

    #unk_emb = UnkEmb()
    #vocab = torchtext.vocab.GloVe(name='6B')
    vocab = nlp_utils.TokenEmbeddings()

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)

    train = squad_reader.load_data(train_reader, tokenizer, limit_count=None)
    test = squad_reader.load_data(test_reader, tokenizer, limit_count=None)
    runtype.adjust_data(train)
    runtype.adjust_data(test)
    print >> sys.stderr, 'Load data over, train=%d, test=%d' % (len(train.qtoks), len(test.qtoks))

    # shuffle training data.
    '''
    print >> sys.stderr, 'begin to shuffle training data..'
    train.shuffle()
    print >> sys.stderr, 'shuffle ok.'
    '''

    # pre-heat.
    preheat(vocab, train.qtoks, train.ctoks, test.qtoks, test.ctoks)

    #check_coverage(train.ctoks, vocab)
    #check_coverage(test.ctoks, vocab)

    # training phase.
    for epoch in range(epoch_count):
        print >> sys.stderr, 'Epoch %d' % epoch
        loss = 0 
        step = 0
        bar = tqdm.tqdm(range(0, len(train.qtoks), batch_size))
        for s in bar:
            optimizer.zero_grad()

            batch_qt = train.qtoks[s:s+batch_size]
            batch_ct = train.ctoks[s:s+batch_size]
            batch_ques_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
            batch_cont_emb = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            # triple output: (batch, clen, 3)
            # binary output: (batch, clen, 2, 2)
            y = train.output[s:s+batch_size]
            y_ = model(batch_ques_emb, batch_cont_emb)
            l = runtype.loss(y_, y)

            step += 1
            loss += l.item()

            l.backward()
            optimizer.step()
            bar.set_description('loss=%.5f' % (loss / step))

        print >> logger, 'epoch=%d\tloss=%.5f' % (epoch, loss/step)

        if test_epoch>0:
            if (epoch+1) % test_epoch ==0:
                print >> logger, 'TestEpoch %d:' % epoch
                train_out = file('log/train.ans.out', 'w')
                test_out = file('log/test.ans.out', 'w')
                run_test(runtype, model, train, batch_size, logger, train_out)
                run_test(runtype, model, test, batch_size, logger, test_out)
                train_out.close()
                test_out.close()
                torch.save(model.state_dict(), opt.save)

    train_out = file('log/train.ans.out', 'w')
    test_out = file('log/test.ans.out', 'w')
    run_test(runtype, model, train, batch_size, logger, train_out)
    run_test(runtype, model, test, batch_size, logger, test_out)
    train_out.close()
    test_out.close()

    # save model.
    torch.save(model.state_dict(), opt.save)

