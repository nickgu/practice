#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch

class V2_MatchAttention_Test(torch.nn.Module):
    '''
    '''
    def __init__(self, emb_size, hidden_size=128, layer_num=1, out_layer_num=2, dropout=0.2):
        super(V2_MatchAttention_Test, self).__init__()

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
        self.__upper_rnn = torch.nn.LSTM(hidden_size*2*3, hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__dense = torch.nn.Sequential(
                torch.nn.Linear(self.__hidden_size*2, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 3)
                )

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

        cq1 = self.cross_feature(c_emb, q_emb)

        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)
        cq_out, _ = self.__rnn(cq1)
    
        # cross q/c
        cq2 = self.cross_feature(c_out, q_out)

        # cat.
        x = torch.cat((c_out, cq2, cq_out), dim=2) # batch, clen, (hidden*2+hidden*2+hidden*2)

        # upper rnn.
        x, _ = self.__upper_rnn(x)
        out = self.__dense(x)
        return out


class V2_MatchAttention_Binary(torch.nn.Module):
    '''
    '''
    def __init__(self, emb_size, hidden_size=512, layer_num=1, out_layer_num=2, dropout=0.4):
        super(V2_MatchAttention_Binary, self).__init__()

        self.__emb_size = emb_size
        self.__hidden_size = hidden_size

        self.__rnn = torch.nn.LSTM(emb_size, hidden_size, 
                #dropout=dropout, 
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

        self.__dense_start = torch.nn.Sequential(
                torch.nn.Linear(self.__hidden_size*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2) )

        self.__rnn_end =  torch.nn.LSTM(hidden_size*2, hidden_size, 
                #dropout=dropout, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True)
        self.__dense_end = torch.nn.Sequential(
                torch.nn.Linear(self.__hidden_size*2*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2) )

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

        out_start = self.__dense_start(x)

        out_end, _ = self.__rnn_end(x)
        out_end = self.__dense_end( torch.cat((x, out_end), dim=2) )
        return torch.cat( (out_start.unsqueeze(2), out_end.unsqueeze(2)), dim=2 )

    def check_gradient(self):
        pass
        '''
        for p in self.__question_rnn.parameters():
            print p.grad
        '''



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


class V2_2_BilinearAttention(torch.nn.Module):
    '''
        ~55% on test 67% one-side (10E)
    '''
    def __init__(self, emb_size, hidden_size=512, layer_num=1, out_layer_num=2, dropout=0.4):
        super(V2_2_BilinearAttention, self).__init__()

        self.__emb_size = emb_size
        self.__hidden_size = hidden_size

        self.__att_w = torch.nn.Parameter(torch.ones(hidden_size*2, hidden_size*2))

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
        batch, clen, _ = c_emb.shape
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(self.__att_w.expand(batch, self.__hidden_size*2, self.__hidden_size*2)).bmm(q_).softmax(dim=2)
        #c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
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

    def check_gradient(self):
        pass
        '''
        for p in self.__question_rnn.parameters():
            print p.grad
        '''


class V2_1_MatchAttention_Binary_BilinearAttention(torch.nn.Module):
    '''
        ~55% on test 67% one-side (10E)
    '''
    def __init__(self, emb_size, hidden_size=512, layer_num=1, out_layer_num=2, dropout=0.4):
        super(V2_1_MatchAttention_Binary_BilinearAttention, self).__init__()

        self.__emb_size = emb_size
        self.__hidden_size = hidden_size

        self.__rnn = torch.nn.LSTM(emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=layer_num,
                #num_layers=self.__layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__att_w = torch.nn.Parameter(torch.ones(hidden_size*2, hidden_size*2))

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
                torch.nn.Linear(self.__hidden_size*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2) )

        self.__dense_end = torch.nn.Sequential(
                torch.nn.Linear(self.__hidden_size*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2) )

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        batch, clen, _ = c_emb.shape
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        c_att_on_q = c_emb.bmm(self.__att_w.expand(batch, self.__hidden_size*2, self.__hidden_size*2)).bmm(q_).softmax(dim=2)
        #c_att_on_q = c_emb.bmm(q_).softmax(dim=2) # batch, clen, qlen
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

        out_start = self.__dense_start(x)
        out_end = self.__dense_end(x)
        return torch.cat( (out_start.unsqueeze(2), out_end.unsqueeze(2)), dim=2 )

    def check_gradient(self):
        pass
        '''
        for p in self.__question_rnn.parameters():
            print p.grad
        '''


class V2_1_BiDafLike(torch.nn.Module):
    '''
        ?
    '''
    def __init__(self, emb_size, hidden_size=256, layer_num=1, out_layer_num=2, dropout=0.4):
        super(V2_1_BiDafLike, self).__init__()

        self.__emb_size = emb_size
        self.__hidden_size = hidden_size

        self.__rnn = torch.nn.LSTM(emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=layer_num, 
                batch_first=True, 
                bidirectional=True)

        # input_size:
        #   hidden * 2 * 2
        #       2: bi-directional 
        #       2: (c_out + cq_attention)
        self.__upper_rnn = torch.nn.LSTM(hidden_size*2*3, hidden_size, 
                dropout=dropout, 
                num_layers=out_layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__fc = torch.nn.Linear(self.__hidden_size*2, self.__hidden_size)

        self.__end_rnn = torch.nn.LSTM(self.__hidden_size, hidden_size, 
                dropout=dropout, 
                num_layers=layer_num, 
                batch_first=True, 
                bidirectional=True)

        self.__fc_out = torch.nn.Linear(self.__hidden_size*3, 3)

    def cross_feature(self, c_emb, q_emb):
        # q_emb: (batch, qlen, emb)
        # c_emb: (batch, clen, emb)
        # output: (batch, clen, emb)
        q_ = q_emb.permute(0, 2, 1) # batch, emb, qlen
        cq_matrix = c_emb.bmm(q_)
        
        cq_att = cq_matrix.softmax(dim=2) # batch, clen, qlen
        cq_emb = torch.bmm(cq_matrix, q_emb)

        # batch, clen
        batch, clen, emb_size = c_emb.shape
        qc_att = cq_matrix.max(dim=2).values.softmax(dim=1)
        qc_emb = qc_att.unsqueeze(1).bmm(c_emb).expand(batch, clen, emb_size)
        return torch.cat( (cq_emb, qc_emb), dim=2 )

    def forward(self, q_emb, c_emb):
        batch = c_emb.shape[0]
        seq_len = c_emb.shape[1] # batch, seq_len, hidden*bi

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

        end_rnn_out, _ = self.__end_rnn(out)
        y = self.__fc_out( torch.cat( (out, end_rnn_out), dim=2 ) )  # batch, clen, (hidden + hidden * 2)
        return y

    def check_gradient(self):
        pass
        '''
        for p in self.__question_rnn.parameters():
            print p.grad
        '''


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
