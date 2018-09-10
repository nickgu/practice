#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
import torch.nn as nn
import tqdm

class CommonDataLoader:
    def __init__(self):
        self.batch_size = 100

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def batch_per_epoch(self):
        pass

    def next_iter(self):
        # genenrate data in variables
        pass


def easy_train(forward_and_backward_fn, data, optimizer, iteration_count=-1, epoch_count=-1):
    if iteration_count > 0:
        process_bar = tqdm.tqdm(range(int(iteration_count)))
    else:
        batch_per_epoch = data.batch_per_epoch()
        batch_count = epoch_count * batch_per_epoch / data.batch_size
        process_bar = tqdm.tqdm(range(int(batch_count)))

    acc_loss = 1.0
    for i in process_bar:
        optimizer.zero_grad()
        cur_loss = forward_and_backward_fn()
        optimizer.step()

        acc_loss = acc_loss * 0.99 + 0.01 * cur_loss
        process_bar.set_description("Loss:%0.3f, AccLoss:%.3f, lr: %0.6f" %
                                    (cur_loss, acc_loss, optimizer.param_groups[0]['lr']))


def easy_test(model, x, y):
    # easy test for softmax-network.
    # the net may design like this:
    #
    #   x_ = ...
    #   x_ = ...
    #   y_ = softmax(self.fc(x_))
    #   loss = torch.nn.CrossEntropy(y_, y)
    #
    #   max(1) : max dim at dim-1
    #   [1] : get dim.
    y_ = model.forward(x).max(1)[1]
    #   check the precision
    hit = y.eq(y_).sum()
    total = len(y)
    import sys
    print >> sys.stderr, ' >>> easy_test_result: %.2f%% (%d/%d) <<<' % (hit*100./total, hit, total)

if __name__=='__main__':
    # test code.
    class LR(nn.Module):
        def __init__(self, in_size):
            nn.Module.__init__(self)
            self.fc = nn.Linear(in_size, 2)

        def forward(self, x):
            import torch.nn.functional as F
            return F.relu(self.fc(x))

    class TrainData(CommonDataLoader):
        def __init__(self, batch_size=100):
            self.batch_size = batch_size
            self.batch_per_epoch = 1000

        def next_iter(self):
            from torch.autograd import Variable

            x = torch.randn(self.batch_size, 2) * 10.
            y = torch.empty(self.batch_size).random_(2).long()
            for idx, a in enumerate(x):
                if torch.cos(a[0]) > a[1]:
                    y[idx] = 0
                else:
                    y[idx] = 1
            
            X = Variable(torch.tensor(x).float())
            Y = Variable(torch.tensor(y).long())
            return X, Y


    import torch.optim as optim

    data = TrainData()
    data.set_batch_size(100)

    model = LR(2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    def fwbp():
        x, y = data.next_iter()
        y_ = model.forward(x)
        loss = loss_fn(y_, y)
        loss.backward()
        return loss[0] / data.batch_size

    # test.
    x, y = data.next_iter()
    easy_test(model, x, y)

    easy_train(fwbp, data, optimizer, iteration_count=1000)

    # test.
    x, y = data.next_iter()
    easy_test(model, x, y)




