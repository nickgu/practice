#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch 
import sys

class V4_MatchAttention_Dropout(torch.nn.Module):
    '''
        Test: 63.46%
        - dropout(input and final linear)
        - add had_cq to g
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, 
            hidden_size=128, layer_num=2, out_layer_num=2, dropout=0.2):

        super(V4_BiDafAdjust, self).__init__()
        self.__hidden_size = hidden_size

        if pretrain_weights is None:
            print >> sys.stderr, 'Init embedding.'
            self.__embed = torch.nn.Embedding(vocab_size, emb_size)
            self.__emb_size = emb_size

        else:
            print >> sys.stderr, 'Init embedding from pretrained.'
            self.__embed = torch.nn.Embedding.from_pretrained(pretrain_weights)
            self.__emb_size = pretrain_weights.shape[1]

        self.__input_dropout = torch.nn.Dropout(dropout)

        self.__rnn = torch.nn.LSTM(self.__emb_size, self.__hidden_size, 
                dropout=dropout, 
                num_layers=layer_num,
                batch_first=True, 
                bidirectional=True)

        self.__g_width = self.__hidden_size * 2 * 3

        # input: g
        self.__rnn_start = torch.nn.LSTM(
                self.__g_width, 
                self.__hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        # input: (g, x_start)
        self.__dense_start = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(self.__hidden_size*2, 1))

        # input: (x_start)
        self.__rnn_end =  torch.nn.LSTM(
                self.__hidden_size*2, 
                self.__hidden_size, 
                dropout=dropout, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True)

        # input: (x_start, x_end)
        self.__dense_end = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(self.__hidden_size*4, 1))

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(c_att_on_q, q_emb)
        return cq_emb

    def forward(self, q_tok_id, c_tok_id):
        q_emb = self.__input_dropout(self.__embed(q_tok_id))
        c_emb = self.__input_dropout(self.__embed(c_tok_id))

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
    
        # cross q/c
        cq = self.cross_feature(c_out, q_out)
        had_cq = c_out * cq

        # cat.
        g = torch.cat((c_out, cq, had_cq), dim=2) # batch, clen, self.__g_width

        # upper rnn.
        x_start, _ = self.__rnn_start(g)
        out_start = self.__dense_start(x_start)

        x_end, _ = self.__rnn_end(x_start)
        out_end = self.__dense_end(torch.cat((x_start, x_end), dim=2))
        return torch.cat( (out_start.permute(0,2,1), out_end.permute(0,2,1)), dim=1 )


class V3_MatchAttention_OutputAdjust(torch.nn.Module):
    '''
        Test: 60.08% (OSM=70.27%)
            - hidden = 300

        Test: 59.59% (OSM=69.83%):
            - Classification on whole output (+ ~1%)
            - make output fc 1-layer (+ ~1%)
    '''
    def __init__(self, vocab_size=None, emb_size=None, 
            pretrain_weights=None, hidden_size=300, layer_num=1, out_layer_num=2, dropout=0.2):
        super(V3_MatchAttention_OutputAdjust, self).__init__()
        self.__hidden_size = hidden_size

        if pretrain_weights is None:
            print >> sys.stderr, 'Init embedding.'
            self.__embed = torch.nn.Embedding(vocab_size, emb_size)
            self.__emb_size = emb_size

        else:
            print >> sys.stderr, 'Init embedding from pretrained.'
            self.__embed = torch.nn.Embedding.from_pretrained(pretrain_weights)
            self.__emb_size = pretrain_weights.shape[1]

        self.__rnn = torch.nn.LSTM(self.__emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=layer_num,
                batch_first=True, 
                bidirectional=True)

        # input_size:
        #   hidden * 2 * 2
        #       2: bi-directional 
        #       2: (c_out + cq_attention)
        self.__upper_rnn = torch.nn.LSTM(hidden_size*2*2, hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__dense_start = torch.nn.Linear(self.__hidden_size*2, 1)

        self.__rnn_end =  torch.nn.LSTM(hidden_size*2, hidden_size, 
                dropout=dropout, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True)

        self.__dense_end = torch.nn.Linear(self.__hidden_size*2*2, 1)

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(c_att_on_q, q_emb)
        return cq_emb

    def forward(self, q_tok_id, c_tok_id):
        q_emb = self.__embed(q_tok_id)
        c_emb = self.__embed(c_tok_id)

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
    
        # cross q/c
        cq = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, cq), dim=2) # batch, clen, (rnn_out_size + emb + hidden*2)

        # upper rnn.
        x_start, _ = self.__upper_rnn(x)
        out_start = self.__dense_start(x_start)

        out_end, _ = self.__rnn_end(x_start)
        out_end = self.__dense_end( torch.cat((x_start, out_end), dim=2) )

        # return shape (batch, 2, clen)
        return torch.cat( (out_start.permute(0,2,1), out_end.permute(0,2,1)), dim=1 )



class V2_MatchAttention_Binary(torch.nn.Module):
    '''
        Test 57% with DP decision.
        Train 71%
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, hidden_size=128, layer_num=1, out_layer_num=2, dropout=0.2):
        super(V2_MatchAttention_Binary, self).__init__()
        self.__hidden_size = hidden_size

        if pretrain_weights is None:
            print >> sys.stderr, 'Init embedding.'
            self.__embed = torch.nn.Embedding(vocab_size, emb_size)
            self.__emb_size = emb_size

        else:
            print >> sys.stderr, 'Init embedding from pretrained.'
            self.__embed = torch.nn.Embedding.from_pretrained(pretrain_weights)
            self.__emb_size = pretrain_weights.shape[1]

        self.__rnn = torch.nn.LSTM(self.__emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=layer_num,
                batch_first=True, 
                bidirectional=True)

        # input_size:
        #   hidden * 2 * 2
        #       2: bi-directional 
        #       2: (c_out + cq_attention)
        self.__upper_rnn = torch.nn.LSTM(hidden_size*2*2, hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__dense_start = torch.nn.Sequential(
                torch.nn.Linear(self.__hidden_size*2, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 2) )

        self.__rnn_end =  torch.nn.LSTM(hidden_size*2, hidden_size, 
                #dropout=dropout, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True)

        self.__dense_end = torch.nn.Sequential(
                torch.nn.Linear(self.__hidden_size*2*2, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 2) )

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(c_att_on_q, q_emb)
        return cq_emb

    def forward(self, q_tok_id, c_tok_id):
        q_emb = self.__embed(q_tok_id)
        c_emb = self.__embed(c_tok_id)

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
    
        # cross q/c
        cq = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, cq), dim=2) # batch, clen, (rnn_out_size + emb + hidden*2)

        # upper rnn.
        x, _ = self.__upper_rnn(x)

        out_start = self.__dense_start(x)

        out_end, _ = self.__rnn_end(x)
        out_end = self.__dense_end( torch.cat((x, out_end), dim=2) )
        return torch.cat( (out_start.unsqueeze(2), out_end.unsqueeze(2)), dim=2 )




class V2_MatchAttention_EmbTrainable(torch.nn.Module):
    '''
        ~42% on test (8E)
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, hidden_size=128, layer_num=1, out_layer_num=2, dropout=0.2):
        super(V2_MatchAttention_EmbTrainable, self).__init__()
        self.__hidden_size = hidden_size

        if pretrain_weights is None:
            print >> sys.stderr, 'Init embedding.'
            self.__embed = torch.nn.Embedding(vocab_size, emb_size)
            self.__emb_size = emb_size

        else:
            print >> sys.stderr, 'Init embedding from pretrained.'
            self.__embed = torch.nn.Embedding.from_pretrained(pretrain_weights)
            self.__emb_size = pretrain_weights.shape[1]

        self.__rnn = torch.nn.LSTM(self.__emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=layer_num,
                batch_first=True, 
                bidirectional=True)

        # input_size:
        #   hidden * 2 * 2
        #       2: bi-directional 
        #       2: (c_out + cq_attention)
        self.__upper_rnn = torch.nn.LSTM(hidden_size*2*2, hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__fc = torch.nn.Linear(self.__hidden_size*2, 128)
        self.__fc_2 = torch.nn.Linear(128, 3)

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(c_att_on_q, q_emb)
        return cq_emb

    def forward(self, q_tok_id, c_tok_id):
        q_emb = self.__embed(q_tok_id)
        c_emb = self.__embed(c_tok_id)

        batch = c_emb.shape[0]
        seq_len = c_emb.shape[1] # batch, seq_len, hidden*bi
        #c1 = self.cross_feature(c_emb, q_emb)

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
    
        # cross q/c
        c2 = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, c2), dim=2) # batch, clen, (rnn_out_size + emb + hidden*2)

        # upper rnn.
        x, _ = self.__upper_rnn(x)

        out = self.__fc(x)
        out = torch.relu(out)
        out = self.__fc_2(out)
        return out

class V2_MatchAttention(torch.nn.Module):
    '''
        ~40% on test (10E)
    '''
    def __init__(self, emb_size, hidden_size=512, layer_num=1, out_layer_num=2, dropout=0.4):
        super(V2_MatchAttention, self).__init__()

        self.__emb_size = emb_size
        self.__hidden_size = hidden_size

        self.__rnn = torch.nn.LSTM(emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=layer_num,
                #num_layers=self.__layer_num, 
                batch_first=True, 
                bidirectional=True)

        # input_size:
        #   hidden * 2 * 2
        #       2: bi-directional 
        #       2: (c_out + cq_attention)
        self.__upper_rnn = torch.nn.LSTM(hidden_size*2*2, hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__fc = torch.nn.Linear(self.__hidden_size*2, 128)
        self.__fc_2 = torch.nn.Linear(128, 3)

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(c_att_on_q, q_emb)
        return cq_emb

    def forward(self, q_emb, c_emb):
        batch = c_emb.shape[0]
        seq_len = c_emb.shape[1] # batch, seq_len, hidden*bi
        #c1 = self.cross_feature(c_emb, q_emb)

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
    
        # cross q/c
        c2 = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, c2), dim=2) # batch, clen, (rnn_out_size + emb + hidden*2)

        # upper rnn.
        x, _ = self.__upper_rnn(x)

        out = self.__fc(x)
        out = torch.relu(out)
        out = self.__fc_2(out)
        return out

if __name__=='__main__':
    pass
