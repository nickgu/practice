#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
import sys
import torch.nn.utils.rnn as rnn_utils
import py3dev
from transformers import *

class V6_Bert(torch.nn.Module):
    '''
        import tokens
    '''
    def __init__(self, model_name='bert-base-uncased'):
        super(V6_Bert, self).__init__()
        py3dev.info('Bert from [%s]' % model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(768, 2)

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

class V5_BiDafAdjust(torch.nn.Module):
    '''
        63.9%
        char-cnn : seems no useful.
        qc-att : seems no useful.
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, 
            hidden_size=128, layer_num=2, out_layer_num=2, dropout=0.2):

        super(V5_BiDafAdjust, self).__init__()
        self.__hidden_size = hidden_size

        print('Init embedding from pretrained.', file=sys.stderr)
        self.__embed = torch.nn.Embedding.from_pretrained(pretrain_weights) #, freeze=False)
        self.__emb_size = pretrain_weights.shape[1]

        self.__input_dropout = torch.nn.Dropout(dropout)

        self.__att_func = torch.nn.Linear(self.__hidden_size*2*2+1, 1)

        self.__rnn = torch.nn.LSTM(self.__emb_size, self.__hidden_size, 
                dropout=dropout, 
                num_layers=layer_num,
                batch_first=True, 
                bidirectional=True)

        self.__g_width = self.__hidden_size * 2 * 6
        self.__m_width = self.__hidden_size * 2

        # input: g
        self.__rnn_cq = torch.nn.LSTM(
                self.__g_width, 
                self.__hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        # input: (g, x_start)
        self.__dense_start = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(self.__m_width, 1))

        # input: (x_start)
        self.__rnn_end =  torch.nn.LSTM(
                self.__m_width, 
                self.__hidden_size, 
                dropout=dropout, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True)

        # input: (x_start, x_end)
        self.__dense_end = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(self.__m_width + self.__hidden_size*2, 1))

    def bidaf_cross(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        batch, clen, emb = c_emb.shape
        qlen = q_emb.shape[1]

        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        cq_dot = c_emb.bmm(q_) # batch, clen, qlen

        # sim_func = dot
        #cq_sim = cq_dot

        # sim_func = att_w * (c, q, dot(c, q))
        c_emb_ = c_emb.unsqueeze(2).expand((batch, clen, qlen, emb))
        q_emb_ = q_emb.unsqueeze(1).expand((batch, clen, qlen, emb))
        cq_dot_ = cq_dot.unsqueeze(3)
        sim_input = torch.cat( (c_emb_, q_emb_, cq_dot_), dim=3 )
        cq_sim = self.__att_func(sim_input).squeeze()
       
        cq_att = cq_sim.softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(cq_att, q_emb)

        qc_att = cq_sim.max(dim=2).values.softmax(dim=1) # batch, clen
        qc_att = qc_att.unsqueeze(1).bmm(c_emb) # batch, 1, emb
        qc_emb = qc_att.expand(batch, clen, emb)

        return cq_emb, qc_emb

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(c_att_on_q, q_emb)
        return cq_emb

    def cq_attention(self, q_out, c_out):
        cq, qc = self.bidaf_cross(c_out, q_out)
        had_cq = c_out * cq
        had_qc = c_out * qc

        cc = self.cross_feature(c_out, c_out)
        ccq = self.cross_feature(c_out, cq)

        g = torch.cat((c_out, cq, had_cq, had_qc, cc, ccq), dim=2) # batch, clen, self.__g_width
        g, _ = self.__rnn_cq(g)
        return g

    ''' for r-net
    def cc_attention(self, c_out):
        cc = self.cross_feature(c_out, c_out)
        had_cc = c_out * cc
        g = torch.cat((c_out, cc, had_cc), dim=2) # batch, clen, self.__g_width
        g, _ = self.__rnn_cc(g)
        return g
    '''

    def forward(self, q_tok_id, c_tok_id):
        q_emb = self.__input_dropout(self.__embed(q_tok_id))
        c_emb = self.__input_dropout(self.__embed(c_tok_id))

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
    
        m = self.cq_attention(q_out, c_out)

        # output layer.
        out_start = self.__dense_start(m)
        x_end, _ = self.__rnn_end(m)
        out_end = self.__dense_end(torch.cat((m, x_end), dim=2))
        return torch.cat( (out_start.permute(0,2,1), out_end.permute(0,2,1)), dim=1 )
