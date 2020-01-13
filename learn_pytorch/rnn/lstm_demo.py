#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

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
    
    batch_size = 100
    length = 64
    vocab_size = 16
    train_x = torch.randint(vocab_size, [batch_size, length])
    train_y = []
    for b in range(batch_size):
        t = []
        for i in range(length):
            if train_x[b][i] == train_x[b][i-1]:
                t.append(1)
            else:
                t.append(0)
        train_y.append(t)
    train_y = torch.tensor(train_y)

    print train_x
    print train_y

    # make model.
    model = RNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    for epoch in range(200):
        print 'Epoch %d' % epoch
        #test_code()

        optimizer.zero_grad()
        y = model(train_x)
        loss = criterion(y.view(-1,2), train_y.view(-1))
        loss.backward()
        print 'loss', loss
        optimizer.step()


    # test code.
    count = 0
    correct = 0
    error = 0
    for test_count in range(100):
        x = torch.randint(vocab_size, [1, length])
        y = model(x).softmax(dim=2).max(dim=2).indices
        print x.tolist()
        print y.tolist()
        last = -1
        for a, b in zip(x.view(-1).tolist(), y.view(-1).tolist()):
            if a == last:
                if b == 1: 
                    correct+=1
                count += 1
            else:
                if b == 1:
                    error+=1
            last = a

    print count, correct, error
    print 'P=%.2f%%, R=%.2f%%' % (correct*100./(correct+error), correct*100./count)





