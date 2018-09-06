#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch

class Trainset:
    def __init__(self):
        self.iter = 0
        self.epoch = 0

    def next_iter(self):
        # genenrate data in variables
        self.iter += 1
 
class CommonTrainer:
    def __init__(self,
                 iteration=10,
                 initial_lr=0.01):
        """Initilize class parameters.

        Args:
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.

        Returns:
            None.
        """
        self.iteration = iteration
        self.initial_lr = initial_lr

        '''
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)
        '''
        self.optimizer = optim.SparseAdam(
            self.skip_gram_model.parameters(), lr=self.initial_lr)


    def train(self, model, data):
        """Multiple training.
        Returns:
            None.
        """
        pair_count = data.evaluate_pair_count(self.window_size)
        process_bar = tqdm(range(int(batch_count)))

        acc_loss = 0.0
        for i in process_bar:
            x1, x2 = data.next_iter()

            self.optimizer.zero_grad()
            loss = self.model.forward(x1, x2)
            loss.backward()
            self.optimizer.step()

            cur_loss = loss.data[0] / len(pos_pairs)
            acc_loss = acc_loss * 0.99 + 0.01 * cur_loss
            process_bar.set_description("Loss:%0.3f, AccLoss:%.3f, lr: %0.6f" %
                                        (cur_loss, acc_loss,
                                         self.optimizer.param_groups[0]['lr']))

if __name__=='__main__':
    pass
