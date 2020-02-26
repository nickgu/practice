#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import time
import squad_reader
import torch
import torchtext 
import torch.nn.utils.rnn as rnn_utils
import sys
import tqdm
import argparse

# import models.
from models import *

sys.path.append('../learn_pytorch')
import nlp_utils

def err(*args):
    print(*args, file=sys.stderr)

def model_params_size(model):
    return sum(x.numel() for x in model.parameters())

class ColorString:
    TC_NONE         ="\033[m"
    TC_RED          ="\033[0;32;31m"
    TC_LIGHT_RED    ="\033[1;31m"
    TC_GREEN        ="\033[0;32;32m"
    TC_LIGHT_GREEN  ="\033[1;32m"
    TC_BLUE         ="\033[0;32;34m"
    TC_LIGHT_BLUE   ="\033[1;34m"
    TC_DARY_GRAY    ="\033[1;30m"
    TC_CYAN         ="\033[0;36m"
    TC_LIGHT_CYAN   ="\033[1;36m"
    TC_PURPLE       ="\033[0;35m"
    TC_LIGHT_PURPLE ="\033[1;35m"
    TC_BROWN        ="\033[0;33m"
    TC_YELLOW       ="\033[1;33m"
    TC_LIGHT_GRAY   ="\033[0;37m"
    TC_WHITE        ="\033[1;37m"

    def __init__(self):
        pass

    @staticmethod
    def colors(s, color):
       return color + s + ColorString.TC_NONE  

    @staticmethod
    def red(s): return ColorString.colors(s, ColorString.TC_RED)

    @staticmethod
    def yellow(s): return ColorString.colors(s, ColorString.TC_YELLOW)

    @staticmethod
    def green(s): return ColorString.colors(s, ColorString.TC_GREEN)

    @staticmethod
    def blue(s): return ColorString.colors(s, ColorString.TC_BLUE)

    @staticmethod
    def cyan(s): return ColorString.colors(s, ColorString.TC_CYAN)


class Arg(object):
    '''
    Sample code:
        ag=Arg()
        ag.str_opt('f', 'file', 'this arg is for file')
        opt = ag.init_arg()
        # todo with opt, such as opt.file
    '''
    def __init__(self, help='Lazy guy, no help'):
        self.is_parsed = False;
        #help = help.decode('utf-8').encode('gb18030')
        self.__parser = argparse.ArgumentParser(description=help)
        self.__args = None;
        #    -l --log 
        self.str_opt('log', 'l', 'logging level default=[error]', meta='[debug|info|error]');
    def __default_tip(self, default_value=None):
        if default_value==None:
            return ''
        return ' default=[%s]'%default_value

    def bool_opt(self, name, iname, help=''):
        #help = help.decode('utf-8').encode('gb18030')
        self.__parser.add_argument(
                '-'+iname, 
                '--'+name, 
                action='store_const', 
                const=1,
                default=0,
                help=help);
        return

    def str_opt(self, name, iname, help='', default=None, meta=None):
        help = (help + self.__default_tip(default))#.decode('utf-8').encode('gb18030')
        self.__parser.add_argument(
                '-'+iname, 
                '--'+name, 
                metavar=meta,
                help=help,
                default=default);
        pass

    def var_opt(self, name, meta='', help='', default=None):
        help = (help + self.__default_tip(default).decode('utf-8').encode('gb18030'))
        if meta=='':
            meta=name
        self.__parser.add_argument(name,
                metavar=meta,
                help=help,
                default=default) 
        pass

    def init_arg(self, input_args=None):
        if not self.is_parsed:
            if input_args is not None:
                self.__args = self.__parser.parse_args(input_args)
            else:
                self.__args = self.__parser.parse_args()
            self.is_parsed = True;
        if self.__args.log:
            format='%(asctime)s %(levelname)8s [%(filename)18s:%(lineno)04d]: %(message)s'
            if self.__args.log=='debug':
                logging.basicConfig(level=logging.DEBUG, format=format)
                logging.debug('log level set to [%s]'%(self.__args.log));
            elif self.__args.log=='info':
                logging.basicConfig(level=logging.INFO, format=format)
                logging.info('log level set to [%s]'%(self.__args.log));
            elif self.__args.log=='error':
                logging.basicConfig(level=logging.ERROR, format=format)
                logging.info('log level set to [%s]'%(self.__args.log));
            else:
                logging.error('log mode invalid! [%s]'%self.__args.log)
        return self.__args

    @property
    def args(self):
        if not self.is_parsed:
            self.__args = self.__parser.parse_args()
            self.is_parsed = True;
        return self.__args;


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

    def loss(self, predict, data, beg, end):
        target = data.binary_output[beg:end]
        batch_context_output = rnn_utils.pad_sequence(target, batch_first=True).detach().cuda()
        l = self.__criterion(predict.view(-1,2), batch_context_output.view([-1]))
        return l

    def get_ans_range(self, y):
        # input_shape: (batch, clen, 2, 2)
        #return y.permute(0,2,1,3)[:,:,:,1:].squeeze().max(dim=2).indices
        prob = y[:,:,:,1:].squeeze()
        return dp_to_generate_answer_range(prob)

    def access(self, y, batch_idx, s0e1, pos):
        return y[batch_idx][pos][s0e1][1]

class RunConfigTriple:
    def __init__(self):
        self.__criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 50., 50.]).cuda())

    def loss(self, predict, data, beg, end):
        target = data.triple_output[beg:end]
        batch_context_output = rnn_utils.pad_sequence(target, batch_first=True).detach().cuda()
        l = self.__criterion(predict.view(-1,3), batch_context_output.view([-1]))
        return l

    def p(self, y):
        #return y.permute((0,2,1)).max(dim=2).indices[:,1:]
        prob = y[:,:,1:]
        return dp_to_generate_answer_range(prob)

class RunConfigSeqBinary:
    def __init__(self):
        self.__criterion = torch.nn.CrossEntropyLoss()

    def loss(self, predict, data, beg, end):
        target = data.answer_range[beg:end]
        batch_context_output = torch.tensor(target).cuda()
        batch = batch_context_output.shape[0]
        l = self.__criterion(predict.view(batch*2,-1), batch_context_output.view([-1]))
        return l

    def get_ans_range(self, y):
        prob = y.permute(0,2,1)
        return dp_to_generate_answer_range(prob)

    def p(self, y, batch_idx, s0e1, pos):
        return y[batch_idx][s0e1][pos]


def run_test(runconfig, model, data, batch_size, logger=None, answer_output=None):
    # test code.
    model.eval()
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
            if runconfig.input_token_id:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            else:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            batch, clen = batch_cont_x.shape[:2]

            # triple output: (batch, clen, 3)
            # binary output: (batch, clen, 2, 2)
            if runconfig.input_char:
                # for char in.
                batch_ques_char_x = rnn_utils.pad_sequence(data.qchar[s:s+batch_size], batch_first=True).detach().cuda()
                batch_cont_char_x = rnn_utils.pad_sequence(data.cchar[s:s+batch_size], batch_first=True).detach().cuda()
                y_ = model(batch_ques_x, batch_cont_x, batch_ques_char_x, batch_cont_char_x)

            elif runconfig.input_acture_len:
                q_acture_len = list(map(lambda x:len(x), batch_qt))
                c_acture_len = list(map(lambda x:len(x), batch_ct))
                y_ = model(batch_ques_x, batch_cont_x, q_acture_len, c_acture_len)

            else:
                y_ = model(batch_ques_x, batch_cont_x)

            l = runconfig.loss(y_, data, s, s+batch_size)
            step += 1
            loss += l.item()

            # seems conflict?
            y_ = y_.softmax(dim=-1)
            ans = runconfig.get_ans_range(y_)
            for idx, ((a,b), (c,d)) in enumerate(zip(ans, data.answer_range[s:s+batch_size])):
                # test one side.
                count += 1

                tag = 'Wr'
                em = False
                if a==c and b==d:
                    em = True
                else:
                    # if answer == answer_candidate, also Exact Match.
                    trim_ans = u''.join(data.ctoks[s+idx][a:b]).replace(u' ', '')
                    for ans_cand in data.answer_candidates[s+idx]:
                        adjust_ans = ans_cand.replace(u' ', '')
                        if adjust_ans == trim_ans:
                            em = True
                if em:
                    exact_match += 1
                    tag = 'EM'

                if a==c or b==d:
                    one_side_match += 1
                    if tag != 'EM':
                        if a==c: tag = 'SM_B'
                        else: tag = 'SM_E'
                if a==c:
                    side_match += 1
                if b==d:
                    side_match += 1

                if answer_output:
                    # for binary.
                    p_y_ = (runconfig.p(y_,idx,0,a)*runconfig.p(y_,idx,1,b)).item()
                    p_y = (runconfig.p(y_,idx,0,c)*runconfig.p(y_,idx,1,d)).item()
                    p_ratio = p_y / p_y_
                    print('%d,%d\t%d,%d\t%s\t%.3f\t%f\t%f' % (
                            a,b,c,d, tag, p_ratio, p_y_, p_y), file=answer_output)

        info = '#(%s) EM=%.2f%% (%d/%d), SM=%.2f%%, OSM=%.2f%% Loss=%.5f' % (
                data.data_name,
                exact_match*100./count, exact_match, count, 
                side_match*50./count, 
                one_side_match*100. / count,
                loss/step)

        err(ColorString.yellow(info))
        if logger:
            print(info, file=logger)

def check_coverage(toks, vocab):
    count = 0
    hit = 0
    for sentence in toks:
        count += len(sentence)
        m = vocab.get_vecs_by_tokens(sentence)
        hit += len(filter(lambda x:x, [i.sum().abs()>1e-5 for i in m]))

    err('Vocab coverage: %.2f%% (%d/%d)' % (hit*100./count, hit, count))

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
    err('Pre-heat over', vocab.cache_size())

def fix_model():
    #np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__=='__main__':
    arg = Arg('SQuAD data training program with pytorch.')
    arg.str_opt('epoch', 'e', default='200')
    arg.str_opt('batch', 'b', default='64')
    arg.str_opt('test_epoch', 't', default='5')
    arg.bool_opt('test_mode', 'T')
    arg.str_opt('logname', 'L', default='<auto>')
    arg.str_opt('comment', 'm', default='')
    arg.str_opt('save', 's', default='params/temp_model.pkl')
    arg.bool_opt('shuffle', 'S')
    arg.bool_opt('continue_training', 'c')
    opt = arg.init_arg()
    fix_model()

    # hyper-param.
    epoch_count = int(opt.epoch)
    batch_size = int(opt.batch)
    test_epoch = int(opt.test_epoch)
    load_size = (None, None)
    if opt.test_mode:
        err('Running in TEST-MODE')
        load_size = (5000, 1000)

    err('epoch=%d' % epoch_count)
    err('test_epoch=%d' % test_epoch)
    err('batch_size=%d' % batch_size)
    err('save_model=%s' % opt.save)
    err('continue_training=%s' % opt.continue_training)

    # === Init Model ===
    
    #unk_emb = UnkEmb()
    #vocab = torchtext.vocab.GloVe(name='6B')
    vocab = nlp_utils.TokenEmbeddings()
    input_emb_size = 300

    #runconfig = RunConfigTriple()
    #runconfig = RunConfigBinary()
    runconfig = RunConfigSeqBinary()

    runconfig.input_token_id = True
    runconfig.input_char = False
    runconfig.input_acture_len = False

    # milestones model.
    model = V5_BiDafAdjust(pretrain_weights=vocab.get_pretrained()).cuda()

    # on testing

    model_name = type(model).__name__
    err(' == INIT_MODEL : %s ==' % (model_name))
    err(' == model_size: ', model_params_size(model), ' ==')

    if opt.logname == '<auto>':
        ts = time.strftime('%Y%m%d_%H:%M:%S',time.localtime(time.time()))
        tag = ''
        if opt.comment:
            tag = '_' + opt.comment
        test_tag = ''
        if opt.test_mode:
            test_tag = 'TEST_'
        logname = 'log/all/%s%s_%s%s.log'% (test_tag, model_name, ts, tag)
    else:
        logname = opt.logname
    logger = open(logname, 'w')
    err('log_filename=%s' % logname)

    if opt.continue_training:
        err('prepare to load previous model.')
        model.load_state_dict(torch.load(opt.save))
        err('load over.')

    # for triple.
    # criterion init in runconfig.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)

    # === Init Data ===
    data_path = '../../dataset/squad1/'
    train_filename = data_path + 'train-v1.1.json'
    test_filename = data_path + 'dev-v1.1.json'

    err('train: [%s]' % train_filename)
    err('test: [%s]' % test_filename)

    tokenizer = nlp_utils.init_tokenizer()

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)

    train = squad_reader.load_data(train_reader, tokenizer, data_name='Train', limit_count=load_size[0], read_char=runconfig.input_char)
    test = squad_reader.load_data(test_reader, tokenizer, data_name='Test', limit_count=load_size[1], read_char=runconfig.input_char)
    err('Load data over, train=%d, test=%d' % (len(train.qtoks), len(test.qtoks)))

    # shuffle training data.
    if opt.shuffle:
        err('begin to shuffle training data..')
        train.shuffle()
        err('shuffle ok.')

    # pre-heat.
    preheat(vocab, train.qtoks, train.ctoks, test.qtoks, test.ctoks)

    #check_coverage(train.ctoks, vocab)
    #check_coverage(test.ctoks, vocab)

    # training phase.
    for epoch in range(epoch_count):
        err('Epoch %d' % epoch)
        loss = 0 
        step = 0
        bar = tqdm.tqdm(range(0, len(train.qtoks), batch_size))
        for s in bar:
            # training mode.
            model.train()
            optimizer.zero_grad()

            batch_qt = train.qtoks[s:s+batch_size]
            batch_ct = train.ctoks[s:s+batch_size]
            if runconfig.input_token_id:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_ids_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            else:
                batch_ques_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_qt], batch_first=True).detach().cuda()
                batch_cont_x = rnn_utils.pad_sequence([vocab.get_vecs_by_tokens(toks) for toks in batch_ct], batch_first=True).detach().cuda()

            # triple output: (batch, clen, 3)
            # binary output: (batch, clen, 2, 2)
            if runconfig.input_char:
                # for char in.
                batch_ques_char_x = rnn_utils.pad_sequence(train.qchar[s:s+batch_size], batch_first=True).detach().cuda()
                batch_cont_char_x = rnn_utils.pad_sequence(train.cchar[s:s+batch_size], batch_first=True).detach().cuda()
                y_ = model(batch_ques_x, batch_cont_x, batch_ques_char_x, batch_cont_char_x)
            elif runconfig.input_acture_len:
                q_acture_len = list(map(lambda x:len(x), batch_qt))
                c_acture_len = list(map(lambda x:len(x), batch_ct))
                y_ = model(batch_ques_x, batch_cont_x, q_acture_len, c_acture_len)

            else:
                y_ = model(batch_ques_x, batch_cont_x)

            l = runconfig.loss(y_, train, s, s+batch_size)

            step += 1
            loss += l.item()

            l.backward()
            optimizer.step()
            bar.set_description('loss=%.5f' % (loss / step))

        print('epoch=%d\tloss=%.5f' % (epoch, loss/step), file=logger)


        # run test on Test each epoch.
        test_out = open('log/ans/test.ans.out', 'w')
        run_test(runconfig, model, test, batch_size, logger, test_out)
        test_out.close()

        # run test on Train each test_epoch or first epoch.
        if test_epoch>0 or epoch == 0:
            if (epoch+1) % test_epoch ==0 or epoch == 0:
                print('TestEpoch %d:' % epoch, file=logger)
                train_out = open('log/ans/train.ans.out', 'w')
                run_test(runconfig, model, train, batch_size, logger, train_out)
                train_out.close()
                err('Try to saving model.')
                torch.save(model.state_dict(), opt.save)
                err('Save ok.')

    train_out = open('log/ans/train.ans.out', 'w')
    test_out = open('log/ans/test.ans.out', 'w')
    run_test(runconfig, model, train, batch_size, logger, train_out)
    run_test(runconfig, model, test, batch_size, logger, test_out)
    train_out.close()
    test_out.close()

    # save model.
    torch.save(model.state_dict(), opt.save)

