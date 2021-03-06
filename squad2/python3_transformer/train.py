#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import time
from bert_squad_reader import SquadReader, BertSquadData, bert_load_data
import torch
import torchtext 
import torch.nn.utils.rnn as rnn_utils
import sys
import tqdm
import py3dev
from transformers import * 

# import models.
from models import *

sys.path.append('../learn_pytorch')
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


def model_params_size(model):
    return sum(x.numel() for x in model.parameters())


class RunConfigRange:
    def __init__(self):
        self.__criterion = torch.nn.CrossEntropyLoss()

    def loss(self, predict, target):
        # target shape: [batch, 2]
        #  second dim: (begin_pos, end_pos)
        batch_context_output = target.cuda()
        batch = batch_context_output.shape[0]
        n = batch_context_output.shape[0]
        l = self.__criterion(predict.reshape(batch*2,-1), batch_context_output.reshape([-1]))
        return l

    def get_ans_range(self, y):
        # y: (batch, 2, clen)

        # for input: [beg:end]
        prob = y.permute(0,2,1)
        return dp_to_generate_answer_range(prob)

        # for input: [beg:end+1]
        #s, e = y.split(1, dim=1)
        #return s.argmax(), e.argmax()

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
            batch_offset = data.context_offset[s:s+batch_size]
            batch_qid = data.qid[s:s+batch_size]
            batch_cand = data.answer_candidates[s:s+batch_size]

            batch_qt = data.qtoks[s:s+batch_size]
            batch_ct = data.ctoks[s:s+batch_size]
            batch_output = data.answer_range[s:s+batch_size]

            batch_x = rnn_utils.pad_sequence(data.x[s:s+batch_size], batch_first=True).cuda()
            batch_y = rnn_utils.pad_sequence(data.y[s:s+batch_size], batch_first=True).cuda()
            batch_token_types = rnn_utils.pad_sequence(data.x_token_types[s:s+batch_size], batch_first=True).cuda()
            batch_mask = rnn_utils.pad_sequence(data.x_mask[s:s+batch_size], batch_first=True).cuda()

            # output: (batch, 2, clen)
            y_ = model(batch_x, token_type_ids=batch_token_types, attention_mask=batch_mask)
            l = runconfig.loss(y_, batch_y)
            step += 1
            loss += l.item()

            # seems conflict?
            y_ = y_.softmax(dim=-1)
            ans = runconfig.get_ans_range(y_)
            for idx, ((a,b), (c,d), offset, qid) in enumerate(zip(ans, batch_y, batch_offset, batch_qid)):
                # test one side.
                count += 1

                # if answer == answer_candidate, also Exact Match.
                ans_text = tokenizer.convert_tokens_to_string(batch_ct[idx][a-offset:b-offset])

                tag = 'Wr'
                em = False
                if a==c and b==d:
                    em = True
                else:
                    trim_ans = ans_text.replace(u' ', '')
                    for ans_cand in batch_cand[idx]:
                        adjust_ans = ans_cand.lower().replace(u' ', '')
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
                    print('%d,%d\t%d,%d\t%s\t%.3f\t%f\t%f\t%s\t%s' % (
                            a,b,c,d, tag, p_ratio, p_y_, p_y, qid, ans_text), file=answer_output)

        info = '#(%s) EM=%.2f%% (%d/%d), SM=%.2f%%, OSM=%.2f%% Loss=%.5f' % (
                data.data_name,
                exact_match*100./count, exact_match, count, 
                side_match*50./count, 
                one_side_match*100. / count,
                loss/step)

        py3dev.info(info)
        if logger:
            print(info, file=logger)

def check_coverage(toks, vocab):
    count = 0
    hit = 0
    for sentence in toks:
        count += len(sentence)
        m = vocab.get_vecs_by_tokens(sentence)
        hit += len(filter(lambda x:x, [i.sum().abs()>1e-5 for i in m]))

    py3dev.info('Vocab coverage: %.2f%% (%d/%d)' % (hit*100./count, hit, count))

def fix_model():
    #np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__=='__main__':
    arg = py3dev.Arg('SQuAD data training program with pytorch.')
    arg.str_opt('model', 'M', default='bert-base-uncased')
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
        py3dev.info('Running in TEST-MODE')
        load_size = (1000, 500)

    py3dev.info('epoch=%d' % epoch_count)
    py3dev.info('test_epoch=%d' % test_epoch)
    py3dev.info('batch_size=%d' % batch_size)
    py3dev.info('save_model=%s' % opt.save)
    py3dev.info('continue_training=%s' % opt.continue_training)

    # === Init Model ===
    runconfig = RunConfigRange()

    runconfig.input_token_id = True
    runconfig.input_acture_len = False

    # milestones model.
    py3dev.info('Preparing models..')
    model = V6_Bert(opt.model).cuda()
    #model = V6_Bert.from_pretrained(opt.model).cuda()

    # on testing

    model_name = type(model).__name__
    py3dev.info(' == INIT_MODEL : %s ==' % (model_name))
    py3dev.info(' == model_size: ', model_params_size(model), ' ==')

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
    py3dev.info('log_filename=%s' % logname)

    if opt.continue_training:
        py3dev.info('prepare to load previous model.')
        model.load_state_dict(torch.load(opt.save))
        py3dev.info('load over.')

    # criterion init in runconfig.
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=30000
    )


    # === Init Data ===
    data_path = '../../dataset/squad1/'
    train_filename = data_path + 'train-v1.1.json'
    test_filename = data_path + 'dev-v1.1.json'

    py3dev.info('train: [%s]' % train_filename)
    py3dev.info('test: [%s]' % test_filename)

    #tokenizer = nlp_utils.init_tokenizer()
    py3dev.info('tokenizer from [%s]' % opt.model)
    if '-cased' in opt.model:
        py3dev.info('tokenizer(%s) do_lower_case=True' % opt.model)
        tokenizer = BertTokenizer.from_pretrained(opt.model, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(opt.model)

    train_reader = SquadReader(train_filename)
    test_reader = SquadReader(test_filename)

    train = bert_load_data(train_reader, tokenizer, data_name='Train', limit_count=load_size[0])
    test = bert_load_data(test_reader, tokenizer, data_name='Test', limit_count=load_size[1])
    py3dev.info('Load data over, train=%d, test=%d' % (len(train.qtoks), len(test.qtoks)))

    # shuffle training data.
    if opt.shuffle:
        py3dev.info('begin to shuffle training data..')
        train.shuffle()
        py3dev.info('shuffle ok.')

    # training phase.
    for epoch in range(epoch_count):
        py3dev.info('Epoch %d' % epoch)
        loss = 0 
        step = 0
        bar = tqdm.tqdm(range(0, len(train.qtoks), batch_size))
        for s in bar:
            # training mode.
            model.train()
            optimizer.zero_grad()

            batch_x = rnn_utils.pad_sequence(train.x[s:s+batch_size], batch_first=True).cuda()
            batch_y = rnn_utils.pad_sequence(train.y[s:s+batch_size], batch_first=True).cuda()
            batch_token_types = rnn_utils.pad_sequence(train.x_token_types[s:s+batch_size], batch_first=True).cuda()
            batch_mask = rnn_utils.pad_sequence(train.x_mask[s:s+batch_size], batch_first=True).cuda()

            # output: (batch, 2, clen)
            y_ = model(batch_x, token_type_ids=batch_token_types, attention_mask=batch_mask)
            l = runconfig.loss(y_, batch_y)

            step += 1
            loss += l.item()

            l.backward()

            # clip grad norm.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
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
                py3dev.info('Try to saving model.')
                torch.save(model.state_dict(), opt.save)
                py3dev.info('Save ok.')

    train_out = open('log/ans/train.ans.out', 'w')
    test_out = open('log/ans/test.ans.out', 'w')
    run_test(runconfig, model, train, batch_size, logger, train_out)
    run_test(runconfig, model, test, batch_size, logger, test_out)
    train_out.close()
    test_out.close()

    # save model.
    torch.save(model.state_dict(), opt.save)

