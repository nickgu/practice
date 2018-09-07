#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
import torch.nn as nn

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


def common_train(model, data, optimizer, batch_size, iteration_count=-1, epoch_count=-1):
    data.set_batch_size(batch_size)

    if iteration > 0:
        process_bar = tqdm(range(int(iteration)))
    else:
        batch_per_epoch = data.batch_per_epoch()
        batch_count = epoch_count * batch_per_epoch / batch_size
        process_bar = tqdm(range(int(batch_count)))

    acc_loss = 0.0
    for i in process_bar:
        x = data.next_iter()

        optimizer.zero_grad()
        loss = model.forward(x)
        loss.backward()
        optimizer.step()

        cur_loss = loss.data[0] / len(batch_size)
        acc_loss = acc_loss * 0.99 + 0.01 * cur_loss
        process_bar.set_description("Loss:%0.3f, AccLoss:%.3f, lr: %0.6f" %
                                    (cur_loss, acc_loss,
                                     optimizer.param_groups[0]['lr']))

if __name__=='__main__':
    # test code.
    class LRModel(nn.Model):
        def __init__(self, in_size):
            nn.Model.__init__(self)
            self.fc = nn.Linear(in_size, 1)
            self.loss = nn.CrossEntropyLoss

        def forward(self, x, y):
            y_ = F.relu(self.fc(x))
            self.loss(y, y_)

    class TrainData(CommonDataLoader):
        def __init__(self, x_size):
            self.x_size = x_size
            self.batch_size = 10
            self.batch_per_epoch = 1000

        def next_iter(self):
            pass

    import torch.optim as optim

    data = TrainData(5)
    model = LRModel(5)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    common_train(model, data, optimizer, batch_size=32, iteration=100)





