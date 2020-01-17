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

class TokenID:
    def __init__(self):
        self.__idx2word = []
        self.__word2idx = {}

    def add(self, term):
        if term not in self.__word2idx:
            self.__word2idx[term] = len(self.__idx2word)
            self.__idx2word.append(term)
        return self.__word2idx[term]

    def get_id(self, term):
        return self.__word2idx.get(term, -1)

    def get_term(self, id):
        if id < 0 or id >= len(self.__idx2word):
            return None
        return self.__idx2word[id]

    def size(self):
        return len(self.__idx2word)

    def __iter__(self):
        for term in self.__idx2word:
            yield term

def token2id(train_reader, test_reader, ider, tokenizer):
    for title, doc in train_reader.iter_doc():
        for token in tokenizer(title):
            ider.add(token)
        for token in tokenizer(doc):
            ider.add(token)

    for qid, q, _, _ in train_reader.iter_question():
        for token in tokenizer(q):
            ider.add(token)

    for title, doc in test_reader.iter_doc():
        for token in tokenizer(title):
            ider.add(token)
        for token in tokenizer(doc):
            ider.add(token)

    for qid, q, _, _ in test_reader.iter_question():
        for token in tokenizer(q):
            ider.add(token)

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(Encoder, self).__init__()

        self.__vocab_size = vocab_size
        self.__emb_size = emb_size
        self.__hidden_size = hidden_size
        self.__layer_num = 3

        self.__emb = torch.nn.Embedding(vocab_size, emb_size)
        self.__question_rnn = torch.nn.LSTM(emb_size, hidden_size, num_layers=self.__layer_num, batch_first=True).cuda()
        self.__context_rnn = torch.nn.LSTM(emb_size, hidden_size, num_layers=self.__layer_num, batch_first=True).cuda()

        #self.__fc = torch.nn.Linear(self.__hidden_size, 3).cuda()
        self.__fc = torch.nn.Linear(self.__hidden_size + self.__hidden_size, 128).cuda()
        self.__fc_2 = torch.nn.Linear(128, 3).cuda()


    def forward(self, question_tokens, context_tokens):
        q_emb = self.__emb(question_tokens).cuda()
        c_emb = self.__emb(context_tokens).cuda()

        _, (q_hidden, q_gate) = self.__question_rnn(q_emb)
        context_out, _ = self.__context_rnn(c_emb, (q_hidden, q_gate))
        #context_out, _ = self.__context_rnn(c_emb)

        seq_len = context_out.shape[1] # batch, seq_len, 3

        # expand last layer.
        c = q_hidden[-1].expand( (seq_len, -1, -1))
        c = c.permute((1, 0, 2))
        c = c.reshape( (-1, seq_len, self.__hidden_size) )
        concat_out = torch.cat((context_out, c), dim=2)
        out = self.__fc(concat_out)
        out = torch.relu(out)
        out = self.__fc_2(out)
        return out

    def check_gradient(self):
        pass
        '''
        for p in self.__fc.parameters():
            print p.grad
        '''

def load_data(reader):
    question_tids = []
    context_tids = []
    output = []
    answer_range = []
    #all_question_tokens = []
    #all_context_tokens = []

    for title, context, qid, question, ans, is_impossible in reader.iter_instance():
        ans_start = -1
        ans_text = 'none'
        if not is_impossible:
            ans_start = ans[0]['answer_start']
            ans_text = ans[0]['text']

        context_tokens = []
        answer_token_begin = -1
        answer_token_end = -1

        if context[ans_start:ans_start+len(ans_text)] == ans_text:
            a = context[:ans_start]
            b = context[ans_start : ans_start + len(ans_text)]
            c = context[ans_start+len(ans_text):]

            context_tokens += tokenizer(a)
            answer_token_begin = len(context_tokens)
            context_tokens += tokenizer(b)
            answer_token_end = len(context_tokens)
            context_tokens += tokenizer(c)
        else:
            context_tokens = tokenizer(context)

        context_tokens.append('<end>')
        question_tokens = tokenizer(question)

        #all_context_tokens.append( context_tokens )
        #all_question_tokens.append( question_tokens )

        # question_tokens, context_tokens, context_output(0,1,2)
        q_ids = []
        c_ids = []
        c_out = []
        for tok in question_tokens:
            q_ids.append( ider.add(tok) )
        for idx, tok in enumerate(context_tokens):
            c_ids.append( ider.add(tok) )
            if idx == answer_token_begin:
                c_out.append(1)
            elif idx == answer_token_end:
                c_out.append(2)
            else:
                c_out.append(0)

        question_tids.append(torch.tensor(q_ids))
        context_tids.append(torch.tensor(c_ids))
        output.append(torch.tensor(c_out))
        answer_range.append( (answer_token_first, answer_token_last) )

    return question_tids, context_tids, output, answer_range

if __name__=='__main__':
    data_path = '../dataset/squad2/'
    train_filename = data_path + 'train-v2.0.json'
    test_filename = data_path + 'dev-v2.0.json'

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    ider = TokenID()
    ider.add('<end>')

    train_reader = squad_reader.SquadReader(train_filename)
    test_reader = squad_reader.SquadReader(test_filename)
    
    '''
    token2id(train_reader, test_reader, ider, tokenizer)
    print ider.size()
    '''

    train_ques_tids, train_cont_tids, train_output, train_answer_range = load_data(train_reader)
    test_ques_tids, test_cont_

    question_tokens = []
    train_context = []
    train_output = []
    train_answer_range = []
    all_question_tokens = []
    all_context_tokens = []

    for title, context, qid, question, ans, is_impossible in train_reader.iter_instance():
        ans_start = -1
        ans_text = 'none'
        if not is_impossible:
            ans_start = ans[0]['answer_start']
            ans_text = ans[0]['text']

        context_tokens = []
        answer_token_first = -1
        answer_token_last = -1

        if context[ans_start:ans_start+len(ans_text)] == ans_text:
            a = context[:ans_start]
            b = context[ans_start : ans_start + len(ans_text)]
            c = context[ans_start+len(ans_text):]

            context_tokens += tokenizer(a)
            answer_token_first = len(context_tokens)
            context_tokens += tokenizer(b)
            answer_token_last = len(context_tokens)-1
            context_tokens += tokenizer(c)
        else:
            context_tokens = tokenizer(context)

        context_tokens.append('<end>')
        question_tokens = tokenizer(question)

        all_context_tokens.append( context_tokens )
        all_question_tokens.append( question_tokens )

        # question_tokens, context_tokens, context_output(0,1,2)
        q_ids = []
        c_ids = []
        c_out = []
        for tok in question_tokens:
            q_ids.append( ider.add(tok) )
        for idx, tok in enumerate(context_tokens):
            c_ids.append( ider.add(tok) )
            if idx == answer_token_first:
                c_out.append(1)
            elif idx-1 == answer_token_last:
                c_out.append(2)
            else:
                c_out.append(0)

        train_question.append(torch.tensor(q_ids))
        train_context.append(torch.tensor(c_ids))
        train_output.append(torch.tensor(c_out))
        train_answer_range.append( (answer_token_first, answer_token_last) )

    print >> sys.stderr, 'load data over (vocab=%d)' % (ider.size())

    # hyper-param.
    epoch_count=500
    batch_size = 32
    input_emb_size = 16
    hidden_size = 32

    # make model.
    model = Encoder(ider.size(), input_emb_size, hidden_size)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 10., 10.]).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    '''
    def make_packed_pad_sequence(data):
        return rnn_utils.pack_padded_sequence(
                rnn_utils.pad_sequence(data, batch_first=True), 
                lengths=(5,2,4), 
                batch_first=True, 
                enforce_sorted=False
            )
    '''

    def test_code():
        # test code.
        with torch.no_grad():
            count = 0
            correct = 0

            for s in range(0, len(train_question), batch_size):
                batch_question_ids = rnn_utils.pad_sequence(train_question[s:s+batch_size], batch_first=True)
                batch_context_ids = rnn_utils.pad_sequence(train_context[s:s+batch_size], batch_first=True)

                y = model(batch_question_ids, batch_context_ids)
                p = y.softmax(dim=2)
                ans = p.permute((0,2,1)).max(dim=2).indices

                for idx, ((a,b), (c,d)) in enumerate(zip(ans[:, 1:].tolist(), train_answer_range[s:s+batch_size])):
                    print a,c,'(%d)'%(a==c),b-1,d,'(%d)'%(b-1==d)
                    print all_context_tokens[idx][a:b]
                    print all_context_tokens[idx][c:d+1]
                    count += 2
                    if a==c:
                        correct += 1
                    if b-1==d:
                        correct += 1

            print 'Precise=%.2f%% (%d/%d)' % (correct*100./count, correct, count)

    for epoch in range(epoch_count):
        print 'Epoch %d' % epoch
        loss = 0 
        step = 0
        bar = tqdm.tqdm(range(0, len(train_question), batch_size))
        for s in bar:
            optimizer.zero_grad()

            batch_question_ids = rnn_utils.pad_sequence(train_question[s:s+batch_size], batch_first=True)
            batch_context_ids = rnn_utils.pad_sequence(train_context[s:s+batch_size], batch_first=True)

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

    test_code()






