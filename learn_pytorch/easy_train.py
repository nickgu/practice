#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import sys
import pydev

import torch
import torch.nn as nn
import tqdm

class IndexCoder:
    def __init__(self):
        self.tags = []
        self.index = {}

    def get(self, key):
        # get the index of key,
        # if not exists, return None
        return self.index.get(key, None)

    def alloc(self, key):
        # get or alloc and index for a key.
        # if not exists, alloc one.
        if key not in self.index:
            idx = len(self.tags)
            self.index[key] = idx
            self.tags.append(key)
            return idx
        return self.index[key]

    def get_code(self, idx):
        return self.tags[idx]

    def save(self, fd):
        for value in self.tags:
            fd.write('%s' % value)

    def load(self, fd):
        self.tags = []
        self.index = {}
        for tag in fd.readlines():
            tag = tag.strip()
            self.index[tag] = len(self.index)
            self.tags.append(tag)

def dump_embeddings(emb, fd):
    for emb in emb.weight:
        print >> fd, ','.join(map(lambda x:str(x), emb.tolist()))

def easy_auc(pred, y, reorder=True):
    import sklearn.metrics as M
    tpr, fpr, threshold = M.roc_curve(y, pred, reorder)
    auc = M.auc(tpr, fpr)
    print >> sys.stderr, pydev.ColorString.yellow(' >>> EASY_AUC_TEST: %.4f (%d items) <<<' % (auc, len(pred)))
    return auc

def easy_train(forward_and_backward_fn, optimizer, iteration_count, loss_curve_output=None):
    process_bar = tqdm.tqdm(range(int(iteration_count)))
    acc_loss = 1.0
    try:
        for iter_num in process_bar:
            optimizer.zero_grad()
            cur_loss = forward_and_backward_fn()
            optimizer.step()

            acc_loss = acc_loss * 0.99 + 0.01 * cur_loss
            if loss_curve_output:
                print >> loss_curve_output, '%d,%.3f,%.3f' % (iter_num, acc_loss, cur_loss)
            process_bar.set_description("Loss:%0.3f, AccLoss:%.3f, lr: %0.6f" %
                                        (cur_loss, acc_loss, optimizer.param_groups[0]['lr']))

    except Exception, e:
        pydev.err(e)
        pydev.err('Training Exception(may be interrupted by control.)')


def easy_test(model, x, y):
    # easy test for multiclass output.
    # the net may design like this:
    #
    #   x_ = ...
    #   x_ = ...
    #   y_ = self.fc(x_)
    #   loss = torch.nn.CrossEntropy(y_, y)
    #
    #   max(1) : max dim at dim-1
    #   [1] : get dim.
    y_ = model.forward(x).max(1)[1]
    #   check the precision
    hit = y.eq(y_).sum()
    total = len(y)
    print >> sys.stderr, pydev.ColorString.red(' >>> EASY_TEST_RESULT: %.2f%% (%d/%d) <<<' % (hit*100./total, hit, total))

if __name__=='__main__':
    # test code.
    class LR(nn.Module):
        def __init__(self, in_size):
            nn.Module.__init__(self)
            self.fc = nn.Linear(in_size, 2)

        def forward(self, x):
            import torch.nn.functional as F
            return F.relu(self.fc(x))

    class TrainData():
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

    easy_train(fwbp, optimizer, 1000)

    # test.
    x, y = data.next_iter()
    easy_test(model, x, y)




