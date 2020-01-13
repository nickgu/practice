#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import squad_reader
import torch
import torchtext 
import torch.nn.utils.rnn as rnn_utils
import sys

class RNN(torch.nn.Module):
    def __init__(self, vocab_size=256, emb_size=16, hidden_size=16, class_num=2):
        super(RNN, self).__init__()

        self.__vocab_size = vocab_size
        self.__emb_size = emb_size
        self.__hidden_size = hidden_size
        self.__class_num = class_num
        self.__layer_num = 1

        self.__emb = torch.nn.Embedding(vocab_size, emb_size)
        self.__rnn = torch.nn.LSTM(emb_size, hidden_size, num_layers=self.__layer_num, batch_first=True)
        self.__fc = torch.nn.Linear(self.__hidden_size, self.__class_num)

    def forward(self, x_id):
        emb = self.__emb(x_id)
        output, _ = self.__rnn(emb)
        return self.__fc(output)

    def check_gradient(self):
        for p in self.__fc.parameters():
            print p.grad

if __name__=='__main__':
    '''
        construct data:
            - sequence of random char (in a-z)
            - label: if the char of previous is the same, label 1.
    '''
    
    train_x = torch.randint(20, [20])

    # make model.
    model = RNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def make_packed_pad_sequence(data):
        return rnn_utils.pack_padded_sequence(
                rnn_utils.pad_sequence(data, batch_first=True), 
                lengths=(5,2,4), 
                batch_first=True, 
                enforce_sorted=False
            )

    def test_code():
        # test code.
        test_size = 1
        batch_question_ids = rnn_utils.pad_sequence(train_question[:test_size], batch_first=True)
        batch_context_ids = rnn_utils.pad_sequence(train_context[:test_size], batch_first=True)

        y = model(batch_question_ids, batch_context_ids)
        p = y.softmax(dim=2)
        print p.permute((0,2,1)).max(dim=2)

        print train_answer_range[:test_size]

    for epoch in range(50):
        print 'Epoch %d' % epoch
        test_code()
        #for s in range(0, len(train_question), batch_size):
        for s in range(0, 1, batch_size):
            optimizer.zero_grad()

            batch_question_ids = rnn_utils.pad_sequence(train_question[s:s+batch_size], batch_first=True)
            batch_context_ids = rnn_utils.pad_sequence(train_context[s:s+batch_size], batch_first=True)

            temp_output = rnn_utils.pad_sequence(train_output[s:s+batch_size], batch_first=True)
            batch_context_output = torch.tensor(temp_output)

            y = model(batch_question_ids, batch_context_ids)

            p = y.softmax(dim=2)
            print p[0][50]
            print p[0][53]
            print batch_context_output[0][50]
            print batch_context_output[0][53]

            y = y.view(-1, 3)
            y_ = torch.randint(3, [32*205])



            #print 'check model gradient'
            #model.check_gradient()

            loss = criterion(y.view(-1,3), batch_context_output.view([-1]))

            loss.backward()
            print 'loss', loss

            #print 'check model gradient'
            #model.check_gradient()
            optimizer.step()

            # test converge.
            #break
        #break








