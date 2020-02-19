#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch
import sys
import torch.nn.utils.rnn as rnn_utils

class V5_BiDafAdjust(torch.nn.Module):
    '''
        63.9%
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, 
            hidden_size=128, layer_num=2, out_layer_num=2, dropout=0.2):

        super(V5_BiDafAdjust, self).__init__()
        self.__hidden_size = hidden_size

        print >> sys.stderr, 'Init embedding from pretrained.'
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



class V4_MatchAttention_PadLSTM(torch.nn.Module):
    '''
        Classification on whole output.
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, hidden_size=128, layer_num=1, out_layer_num=2, dropout=0.2):
        super(V4_MatchAttention_PadLSTM, self).__init__()
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
                torch.nn.Linear(256, 1) )

        self.__rnn_end =  torch.nn.LSTM(hidden_size*2, hidden_size, 
                dropout=dropout, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True)

        self.__dense_end = torch.nn.Sequential(
                torch.nn.Linear(self.__hidden_size*2*2, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1) )

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(c_att_on_q, q_emb)
        return cq_emb

    def forward(self, q_tok_id, c_tok_id, q_acture_len, c_acture_len):
        def call_lstm(lstm, x, x_length):
            x_ = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)
            y, _ = lstm(x_)
            y, yl = rnn_utils.pad_packed_sequence(y, batch_first=True)
            return y, _

        q_emb = self.__embed(q_tok_id)
        c_emb = self.__embed(c_tok_id)

        q_out, _ = call_lstm(self.__rnn, q_emb, q_acture_len)
        c_out, _ = call_lstm(self.__rnn, c_emb, c_acture_len)
    
        # cross q/c
        cq = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, cq), dim=2) # batch, clen, (rnn_out_size + emb + hidden*2)

        # upper rnn.
        x, _ = call_lstm(self.__upper_rnn, x, c_acture_len)

        out_start = self.__dense_start(x)

        out_end, _ = self.__rnn_end(x)
        out_end = self.__dense_end( torch.cat((x, out_end), dim=2) )

        # return shape (batch, 2, clen)
        return torch.cat( (out_start.permute(0,2,1), out_end.permute(0,2,1)), dim=1 )




class V3_CharCNN_MatAtt(torch.nn.Module):
    '''
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, hidden_size=128, layer_num=1, out_layer_num=2, dropout=0.2):
        super(V3_CharCNN_MatAtt, self).__init__()
        self.__hidden_size = hidden_size

        if pretrain_weights is None:
            print >> sys.stderr, 'Init embedding.'
            self.__embed = torch.nn.Embedding(vocab_size, emb_size)
            self.__emb_size = emb_size

        else:
            print >> sys.stderr, 'Init embedding from pretrained.'
            self.__embed = torch.nn.Embedding.from_pretrained(pretrain_weights)
            self.__emb_size = pretrain_weights.shape[1]


        self.__token_L = 16
        self.__char_emb_size = 16
        self.__char_emb_outsize = 32
        self.__charemb = torch.nn.EmbeddingBag(70001, self.__char_emb_outsize, mode='sum')

        self.__rnn = torch.nn.LSTM(self.__emb_size+self.__char_emb_outsize, hidden_size, 
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

    def forward(self, q_tok_id, c_tok_id, q_char_x, c_char_x):
        qlen = q_char_x.shape[1]
        batch, clen, tok_len = c_char_x.shape

        q_char_emb = self.__charemb(q_char_x.reshape(batch*qlen, -1)).reshape(batch, qlen, -1)
        #q_char_emb = self.__charcnn(q_char_emb.reshape(batch*qlen, 1, tok_len, self.__char_emb_size)).reshape(batch, qlen, -1)
        c_char_emb = self.__charemb(c_char_x.reshape(batch*clen, -1)).reshape(batch, clen, -1)
        #c_char_emb = self.__charemb(c_char_x)
        #c_char_emb = self.__charcnn(c_char_emb.reshape(batch*clen, 1, tok_len, self.__char_emb_size)).reshape(batch, clen, -1)

        q_emb = self.__embed(q_tok_id)
        c_emb = self.__embed(c_tok_id)

        q_out, _ = self.__rnn(torch.cat((q_emb, q_char_emb), dim=2))
        c_out, _ = self.__rnn(torch.cat((c_emb, c_char_emb), dim=2))
    
        # cross q/c
        c2 = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, c2), dim=2) # batch, clen, (rnn_out_size + emb + hidden*2)

        # upper rnn.
        x, _ = self.__upper_rnn(x)

        out_start = self.__dense_start(x)

        out_end, _ = self.__rnn_end(x)
        out_end = self.__dense_end( torch.cat((x, out_end), dim=2) )
        return torch.cat( (out_start.unsqueeze(2), out_end.unsqueeze(2)), dim=2 )


class V3_DropoutMatchAttention(torch.nn.Module):
    '''
        Test 57% with DP decision.
        Train 71%
    '''
    def __init__(self, vocab_size=None, emb_size=None, pretrain_weights=None, hidden_size=128, layer_num=1, out_layer_num=2, dropout=0.2):
        super(V3_DropoutMatchAttention, self).__init__()
        self.__hidden_size = hidden_size

        if pretrain_weights is None:
            print >> sys.stderr, 'Init embedding.'
            self.__embed = torch.nn.Embedding(vocab_size, emb_size)
            self.__emb_size = emb_size

        else:
            print >> sys.stderr, 'Init embedding from pretrained.'
            self.__embed = torch.nn.Embedding.from_pretrained(pretrain_weights)
            self.__emb_size = pretrain_weights.shape[1]

        self.__dropout = torch.nn.Dropout(dropout)

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

        batch = c_emb.shape[0]
        seq_len = c_emb.shape[1] # batch, seq_len, hidden*bi

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
    
        # cross q/c
        c2 = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, c2), dim=2) # batch, clen, (rnn_out_size + emb + hidden*2)
        x = self.__dropout(x)

        # upper rnn.
        x, _ = self.__upper_rnn(x)

        out_start = self.__dense_start(x)

        out_end, _ = self.__rnn_end(x)
        out_end = self.__dense_end( torch.cat((x, out_end), dim=2) )
        return torch.cat( (out_start.unsqueeze(2), out_end.unsqueeze(2)), dim=2 )


#################################################################################################################3

class V0_Encoder(torch.nn.Module):
    '''
        emb->lstm->concat(question_emb, context_emb)->fc x 2
        
        this model integrate wordemb together.
        all_data:
            train : 78%
            test : ~12% 
    '''
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(V0_Encoder, self).__init__()

        self.__vocab_size = vocab_size
        self.__emb_size = emb_size
        self.__hidden_size = hidden_size
        self.__layer_num = 3

        self.__emb = torch.nn.Embedding(vocab_size, emb_size)
        self.__question_rnn = torch.nn.LSTM(emb_size, hidden_size, num_layers=self.__layer_num, batch_first=True).cuda()
        self.__context_rnn = torch.nn.LSTM(emb_size, hidden_size, num_layers=self.__layer_num, batch_first=True).cuda()

        #self.__fc = torch.nn.Linear(self.__hidden_size, 3).cuda()
        self.__fc = torch.nn.Linear(self.__hidden_size + self.__hidden_size, 128).cuda()
        self.__fc_2 = torch.nn.Linear(128, 3).cuda()


    def forward(self, question_tokens, context_tokens):
        q_emb = self.__emb(question_tokens).cuda()
        c_emb = self.__emb(context_tokens).cuda()

        _, (q_hidden, q_gate) = self.__question_rnn(q_emb)
        context_out, _ = self.__context_rnn(c_emb, (q_hidden, q_gate))
        #context_out, _ = self.__context_rnn(c_emb)

        seq_len = context_out.shape[1] # batch, seq_len, 3

        # expand last layer.
        c = q_hidden[-1].expand( (seq_len, -1, -1))
        c = c.permute((1, 0, 2))
        c = c.reshape( (-1, seq_len, self.__hidden_size) )
        concat_out = torch.cat((context_out, c), dim=2)
        out = self.__fc(concat_out)
        out = torch.relu(out)
        out = self.__fc_2(out)
        return out

    def check_gradient(self):
        pass
        '''
        for p in self.__fc.parameters():
            print p.grad
        '''

class V1_CatLstm(torch.nn.Module):
    '''
        lstm->concat(question_emb, context_emb)->fc x 2
        
        this model input wordemb and will not update it.
    '''
    def __init__(self, emb_size, hidden_size, layer_num=3, dropout=0.2):
        super(V1_CatLstm, self).__init__()

        self.__emb_size = emb_size
        self.__hidden_size = hidden_size
        self.__layer_num = layer_num

        self.__rnn = torch.nn.LSTM(emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=self.__layer_num, 
                batch_first=True, 
                bidirectional=True).cuda()
        #self.__context_rnn = torch.nn.LSTM(emb_size, hidden_size, dropout=dropout, num_layers=self.__layer_num, batch_first=True).cuda()

        self.__fc = torch.nn.Linear(self.__hidden_size*2 + self.__hidden_size, 128).cuda()
        self.__fc_2 = torch.nn.Linear(128, 3).cuda()

    def forward(self, q_emb, c_emb):
        #_, (q_hidden, q_gate) = self.__question_rnn(q_emb)
        #context_out, _ = self.__context_rnn(c_emb)

        _, (q_hidden, q_gate) = self.__rnn(q_emb)
        context_out, _ = self.__rnn(c_emb)

        seq_len = context_out.shape[1] # batch, seq_len, hidden*bi

        # expand last layer.
        c = q_hidden[-1].expand( (seq_len, -1, -1))
        c = c.permute((1, 0, 2))
        c = c.reshape( (-1, seq_len, self.__hidden_size) )
        concat_out = torch.cat((context_out, c), dim=2)
        out = self.__fc(concat_out)
        out = torch.relu(out)
        out = self.__fc_2(out)
        return out

    def check_gradient(self):
        for p in self.__question_rnn.parameters():
            print p.grad

class V3_CrossConv(torch.nn.Module):
    '''
        not yet complete.
    '''
    def __init__(self):
        super(V3_CrossConv, self).__init__()

        self.__Q_topk = 5
        self.__conv_out_size=256

        self.__conv_stack = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, self.__conv_out_size, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                )


        self.__fc = torch.nn.Linear(self.__Q_topk * self.__conv_out_size, 128)
        self.__fc2 = torch.nn.Linear(128, 3)
        
    def forward(self, q_emb, c_emb):
        batch = c_emb.shape[0]
        clen = c_emb.shape[1]

        # bmm embeddings to get an similarity map.
        cq_bmm = c_emb.bmm(q_emb.permute(0,2,1)).topk(k=self.__Q_topk, dim=2).values

        # batch, clen, emb+qlen
        x = torch.cat((c_emb, cq_bmm), dim=2)

        x = cq_bmm.unsqueeze(1)
        x = self.__conv_stack(x) # batch, chan, clen, emb+topk_sim

        # permute and flatten
        x = x.permute(0,2,1,3)
        x = x.reshape(batch, clen, -1) # batch, clen,  Q_topk * chan

        out = self.__fc(x)
        out = torch.relu(out)
        out = self.__fc2(out)
        return out

    def check_gradient(self):
        pass
        '''
        for p in self.__question_rnn.parameters():
            print p.grad
        '''

class V4_Transformer(torch.nn.Module):
    def __init__(self, emb_size):
        super(V4_Transformer, self).__init__()

        self.__sentence_emb = 32

        self.__type_q_emb = torch.autograd.Variable(torch.randn(self.__sentence_emb))
        self.__type_c_emb = torch.autograd.Variable(torch.randn(self.__sentence_emb))

        self.__dmodel = emb_size + self.__sentence_emb
        layer = torch.nn.TransformerEncoderLayer(d_model=self.__dmodel, nhead=8)
        self.__transformer = torch.nn.TransformerEncoder(layer, num_layers=4)

        self.__fc = torch.nn.Linear(self.__dmodel, 128)
        self.__fc2 = torch.nn.Linear(128, 3)
        
    def forward(self, q_emb, c_emb):
        batch = c_emb.shape[0]
        qlen = q_emb.shape[1]
        clen = c_emb.shape[1]

        qe = self.__type_q_emb.expand(batch, qlen, self.__sentence_emb).cuda()
        ce = self.__type_c_emb.expand(batch, clen, self.__sentence_emb).cuda()
        
        q = torch.cat((q_emb, qe), dim=2)
        c = torch.cat((c_emb, ce), dim=2)

        x = torch.cat((q, c), dim=1)
        x = self.__transformer(x)
        x = x[:,qlen:,:]

        out = self.__fc(x)
        out = torch.relu(out)
        out = self.__fc2(out)
        return out

if __name__=='__main__':
    pass
