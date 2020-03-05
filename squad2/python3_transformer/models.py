#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
import sys
import torch.nn.utils.rnn as rnn_utils
import py3dev
from transformers import *

class V6_Bert(BertPreTrainedModel):
    '''
        import tokens
    '''
    def __init__(self, config):
        super(V6_Bert, self).__init__(config)
        #py3dev.info('Bert from [%s]' % model_name)
        self.bert = BertModel(config)
        self.linear = torch.nn.Linear(768, 2)

        self.init_weights()

    def forward(self, merge_tok_ids, token_type_ids, attention_mask):
        # merge_tok_ids:
        #   [CLS], q1, q2, ..., [SEP], c1, c2, ...
        # token_type_ids:
        #   [0]...[1]...
        # attention_mask:
        #   [1] on valid ids

        batch, clen = merge_tok_ids.shape

        tok_emb, all_emb = self.bert(merge_tok_ids, 
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)

        out = self.linear(tok_emb)
        '''
        out = out * token_type_ids.unsqueeze(2).expand((batch, clen, 2)) \
                * attention_mask.unsqueeze(2).expand((batch, clen, 2))
        '''
        return out.permute(0,2,1)


