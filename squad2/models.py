#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch

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

class V2_MatchAttention(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, layer_num=2, dropout=0.2):
        super(V2_MatchAttention, self).__init__()

        self.__emb_size = emb_size
        self.__hidden_size = hidden_size
        self.__layer_num = layer_num

        self.__rnn = torch.nn.LSTM(emb_size, hidden_size, 
                dropout=dropout, 
                num_layers=self.__layer_num, 
                batch_first=True, 
                bidirectional=True).cuda()

        # input_size:
        #   hidden * 2 * 2
        #       2: bi-directional 
        #       2: (c_out + cq_attention)
        self.__cross_rnn = torch.nn.LSTM(hidden_size*2*2, hidden_size, 
                dropout=dropout, 
                num_layers=self.__layer_num, 
                batch_first=True, 
                bidirectional=True).cuda()

        self.__fc = torch.nn.Linear(self.__hidden_size*2, 128).cuda()
        self.__fc_2 = torch.nn.Linear(128, 3).cuda()

    def forward(self, q_emb, c_emb):
        q_out, _ = self.__rnn(q_emb)
        c_out, _ = self.__rnn(c_emb)

        seq_len = c_out.shape[1] # batch, seq_len, hidden*bi

        # get attention of each context_token on question.
        q_att = q_out.permute(0, 2, 1) # batch, emb, qlen
        cq_att = c_out.bmm(q_att).softmax(dim=2) # batch, clen, qlen

        # (batch, clen, qlen) x (batch, qlen, emb) => (batch, clen, emb)
        # add weighted q_emb to context.
        cq_emb = torch.bmm(cq_att, q_out) 

        cat_c_emb = torch.cat((c_out, cq_emb), dim=2) # batch, clen, (rnn_out_size + emb)
        c_final_output, _ = self.__cross_rnn(cat_c_emb)

        out = self.__fc(c_final_output)
        out = torch.relu(out)
        out = self.__fc_2(out)
        return out

    def check_gradient(self):
        pass
        '''
        for p in self.__question_rnn.parameters():
            print p.grad
        '''

class V3_CrossConv(torch.nn.Module):
    def __init__(self):
        super(V3_CrossConv, self).__init__()

        self.__Q_topk = 5

        self.__conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.__conv2 = torch.nn.Conv2d(8, 16, kernel_size=5, padding=2)

        self.__fc = torch.nn.Linear(self.__Q_topk * 16, 128)
        self.__fc2 = torch.nn.Linear(128, 3)
        
    def forward(self, q_emb, c_emb):
        batch = c_emb.shape[0]
        clen = c_emb.shape[1]

        # bmm embeddings to get an similarity map.
        cq_bmm = c_emb.bmm(q_emb.permute(0,2,1))

        x = cq_bmm.unsqueeze(1)
        x = self.__conv1(x) # batch, chan, clen, qlen
        x = self.__conv2(x) # batch, chan, clen, qlen

        # permute and flatten
        x = x.permute(0,2,1,3)
        x = x.topk(k=self.__Q_topk, dim=3).values
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



if __name__=='__main__':
    pass
