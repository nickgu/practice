#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
import torchtext 
import torch.nn.utils.rnn as rnn_utils
import sys
import tqdm

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
        self.__rnn.cuda()
        self.__fc = torch.nn.Linear(self.__hidden_size, self.__class_num)
        self.__fc.cuda()

    def forward(self, x_id):
        emb = self.__emb(x_id)
        emb = emb.cuda()
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
    cuda = torch.device('cuda')     # Default CUDA device
    
    if len(sys.argv)<2:
        print '%s <epoch_count>' % (sys.argv[0])
        sys.exit(-1)

    epoch_count = int(sys.argv[1])
    print 'epoch=%d' % epoch_count
    data_size = 500
    batch_size = 16
    length = 64
    vocab_size = 32
    train_x = torch.randint(vocab_size, [data_size, length])
    train_y = []
    for b in range(data_size):
        t = []
        for i in range(length):
            if train_x[b][i] == train_x[b][i-1]:
                t.append(1)
            else:
                t.append(0)
        train_y.append(t)
    train_y = torch.tensor(train_y)

    # make model.
    model = RNN()
    #model.to(cuda)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_set = torch.utils.data.TensorDataset(train_x, train_y) 
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

    print 'load data over'
    for epoch in range(epoch_count):
        print 'Epoch %d' % epoch
        all_loss = 0
        batch_count = 0
        bar = tqdm.tqdm(data_loader)
        for x, y in bar:
            optimizer.zero_grad()
            y = y.cuda()
            y_ = model(x)
            loss = criterion(y_.view(-1,2), y.view(-1))
            loss.backward()
            all_loss += loss
            batch_count += 1
            bar.set_description('loss=%f' % (all_loss / batch_count) )
            optimizer.step()

    # test code.
    with torch.no_grad():
        # test original
        # seems that 30 epoch can make convergence (Adam(0.01))
        train_y_ = model(train_x)
        train_y_ = train_y_.view(-1,2).max(dim=1).indices
        pred_label_pair = zip(train_y_.tolist(), train_y.view(-1).tolist())

        count = len(filter(lambda x:x[1]==1, pred_label_pair))
        correct = len(filter(lambda x:(x[0]==x[1] and x[1]==1), pred_label_pair))
        error = len(filter(lambda x:(x[0]==1 and x[1]==0), pred_label_pair))
        print 'train_data:'
        print count, correct, error
        print 'P=%.2f%%, R=%.2f%%' % (correct*100./(correct+error+1e-5), correct*100./count)

        count = 0
        correct = 0
        error = 0
        # test random.
        for test_count in range(100):
            x = torch.randint(vocab_size, [1, length])
            y = model(x).softmax(dim=2).max(dim=2).indices
            #print x.tolist()
            #print y.tolist()
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

        print 'random_data:'
        print count, correct, error
        print 'P=%.2f%%, R=%.2f%%' % (correct*100./(correct+error+1e-5), correct*100./count)





