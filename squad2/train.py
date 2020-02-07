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

def dp_to_generate_answer_range(data):
    ''' 
        data shape: (batch, clen, 2), 
        last dim indicates start/end prob.
    '''
    ans = []
    l = data.shape[1]
    data = data.cpu().numpy()
    dp = [0.] * (l+1)
    dp_sidx = [-1] * (l+1)
    for b in data:
        max_prob = 0
        max_range = (0, 0)
        dp[0] = 0
        dp_sidx[0] = -1
        for idx in range(l):
            sp, ep = b[idx]
            cur_end_prob = dp[idx] * ep
            if cur_end_prob > max_prob:
                max_prob = cur_end_prob
                max_range = (dp_sidx[idx], idx)

            if sp>dp[idx]:
                dp[idx+1] = sp
                dp_sidx[idx+1] = idx
            else:
                dp[idx+1] = dp[idx]
                dp_sidx[idx+1] = dp_sidx[idx]
        ans.append(max_range)
    return ans

class RunConfigBinary:
    def __init__(self):
        self.__criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 100.]).cuda())

    def loss(self, predict, target):
        batch_context_output = rnn_utils.pad_sequence(target, batch_first=True).detach().cuda()
        l = self.__criterion(predict.view(-1,2), batch_context_output.view([-1]))
        return l

    def get_ans_range(self, y):
        # input_shape: (batch, clen, 2, 2)
        #return y.permute(0,2,1,3)[:,:,:,1:].squeeze().max(dim=2).indices
        prob = y[:,:,:,1:].squeeze()
        return dp_to_generate_answer_range(prob)

    def adjust_data(self, data):
        data.output = data.binary_output
          

class RunConfigTriple:
    def __init__(self):
        self.__criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 50., 50.]).cuda())

    def loss(self, predict, target):
        batch_context_output = rnn_utils.pad_sequence(target, batch_first=True).detach().cuda()
        l = self.__criterion(predict.view(-1,3), batch_context_output.view([-1]))
        return l

    def get_ans_range(self, y):
        #return y.permute((0,2,1)).max(dim=2).indices[:,1:]
        prob = y[:,:,1:]
        return dp_to_generate_answer_range(prob)

    def adjust_data(self, data):
        data.output = data.triple_output


def run_test(runconfig, model, data, batch_size, logger=None, answer_output=None):
    # test code.
    with torch.no_grad():
        count = 0
        exact_match = 0
        side_match = 0
        one_side_match = 0
        
        loss = 0 
        step = 0

        for s in tqdm.tqdm(range(0, len(data.qtoks), batch_size)):
            batch_qt = data.qtoks[s:s+batch_size]
            batch_ct = data.ctoks[s:s+batch_size]
            if runconfig.embedding_trainable:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            else:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            batch, clen = batch_cont_x.shape[:2]
            y = rnn_utils.pad_sequence(data.output[s:s+batch_size], batch_first=True)

            # triple output: (batch, clen, 3)
            # binary output: (batch, clen, 2, 2)
            y_ = model(batch_ques_x, batch_cont_x)

            l = runconfig.loss(y_, y)
            step += 1
            loss += l.item()

            ans = runconfig.get_ans_range(y_)
            for idx, ((a,b), (c,d)) in enumerate(zip(ans, data.answer_range[s:s+batch_size])):
                if answer_output:
                    print >> answer_output, '%d,%d\t%d,%d\t%d' % (a,b,c,d, len(data.ctoks[s+idx]))
                # test one side.
                count += 1

                em = False
                if a==c and b==d:
                    em = True
                else:
                    # if answer == answer_candidate, also Exact Match.
                    trim_ans = u''.join(data.ctoks[s+idx][a:b]).replace(u' ', '')
                    for a in data.answer_candidates[s+idx]:
                        adjust_ans = a.replace(u' ', '')
                        if adjust_ans == trim_ans:
                            em = True

                if em:
                    exact_match += 1

                if a==c or b==d:
                    one_side_match += 1
                if a==c:
                    side_match += 1
                if b==d:
                    side_match += 1

        info = '#(%s) EM=%.2f%% (%d/%d), SM=%.2f%%, OSM=%.2f%% Loss=%.5f' % (
                data.data_name,
                exact_match*100./count, exact_match, count, 
                side_match*50./count, 
                one_side_match*100. / count,
                loss/step)

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
    arg.bool_opt('test_mode', 'T')
    opt = arg.init_arg()

    logger = file(opt.logname, 'w')

    # hyper-param.
    epoch_count = int(opt.epoch)
    batch_size = int(opt.batch)
    test_epoch = int(opt.test_epoch)
    load_size = (None, None)
    if opt.test_mode:
        print >> sys.stderr, 'Running in TEST-MODE'
        load_size = (5000, 1000)

    print >> sys.stderr, 'epoch=%d' % epoch_count
    print >> sys.stderr, 'test_epoch=%d' % test_epoch
    print >> sys.stderr, 'batch_size=%d' % batch_size
    print >> sys.stderr, 'log_filename=%s' % opt.logname
    print >> sys.stderr, 'save_model=%s' % opt.save
    print >> sys.stderr, 'continue_training=%s' % opt.continue_training

    # === Init Model ===
    
    #unk_emb = UnkEmb()
    #vocab = torchtext.vocab.GloVe(name='6B')
    vocab = nlp_utils.TokenEmbeddings()
    input_emb_size = 400

    #runconfig = RunConfigTriple()
    runconfig = RunConfigBinary()
    #runconfig.embedding_trainable = False
    runconfig.embedding_trainable = True

    # milestones model.
    #model = V2_MatchAttention(input_emb_size).cuda()
    #model = V2_MatchAttention_EmbTrainable(pretrain_weights=vocab.get_pretrained()).cuda()

    # research model.
    #model = V3_Model(pretrain_weights=vocab.get_pretrained()).cuda()
    #model = V2_MatchAttention_Test(input_emb_size).cuda()
    model = V2_MatchAttention_Binary(pretrain_weights=vocab.get_pretrained()).cuda()
    #model = V0_Encoder(ider.size(), input_emb_size, hidden_size)
    #model = V1_CatLstm(input_emb_size, hidden_size, layer_num=layer_num, dropout=0.4)
    #model = V3_FCEmbModel(input_emb_size).cuda()
    #model = V2_2_BilinearAttention(input_emb_size).cuda()
    #model = V2_1_BiDafLike(input_emb_size).cuda()
    #model = V3_CrossConv().cuda()
    #model = V4_Transformer(input_emb_size).cuda()

    print >> sys.stderr, ' == INIT_MODEL : %s ==' % (type(model).__name__)
    print >> sys.stderr, ' == model_size: ', easy_train.model_params_size(model), ' =='

    if opt.continue_training:
        print >> sys.stderr, 'prepare to load previous model.'
        model.load_state_dict(torch.load(opt.save))
        print >> sys.stderr, 'load over.'

    # for triple.
    # criterion init in runconfig.
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

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)

    train = squad_reader.load_data(train_reader, tokenizer, data_name='Train', limit_count=load_size[0])
    test = squad_reader.load_data(test_reader, tokenizer, data_name='Test', limit_count=load_size[1])
    runconfig.adjust_data(train)
    runconfig.adjust_data(test)
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
            if runconfig.embedding_trainable:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            else:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            # triple output: (batch, clen, 3)
            # binary output: (batch, clen, 2, 2)
            y = train.output[s:s+batch_size]
            y_ = model(batch_ques_x, batch_cont_x)
            l = runconfig.loss(y_, y)

            step += 1
            loss += l.item()

            l.backward()
            optimizer.step()
            bar.set_description('loss=%.5f' % (loss / step))

        print >> logger, 'epoch=%d\tloss=%.5f' % (epoch, loss/step)

        if test_epoch>0:
            if (epoch+1) % test_epoch ==0:
                print >> logger, 'TestEpoch %d:' % epoch
                train_out = file('log/ans/train.ans.out', 'w')
                test_out = file('log/ans/test.ans.out', 'w')
                run_test(runconfig, model, train, batch_size, logger, train_out)
                run_test(runconfig, model, test, batch_size, logger, test_out)
                train_out.close()
                test_out.close()
                torch.save(model.state_dict(), opt.save)

    train_out = file('log/ans/train.ans.out', 'w')
    test_out = file('log/ans/test.ans.out', 'w')
    run_test(runconfig, model, train, batch_size, logger, train_out)
    run_test(runconfig, model, test, batch_size, logger, test_out)
    train_out.close()
    test_out.close()

    # save model.
    torch.save(model.state_dict(), opt.save)

