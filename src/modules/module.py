# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


from src.components import operation as op
from src.components.layers import  EmbeddingCollection, LSTMEncoder, SelfAttention, MLPAttention,FG_fusion

MASK_VALUE = -2 ** 32 + 1

class ModelManager(nn.Module):

    def __init__(self, args, num_char, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        # hyper-parameters
        self.__num_char = num_char
        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # Initialize an char embedding object.
        self.__char_embedding = EmbeddingCollection(self.__num_char, self.__args.char_embedding_dim)

        # Initialize an word embedding object.
        self.__word_embedding = EmbeddingCollection(self.__num_word, self.__args.word_embedding_dim)

        # TODO: Now, output dim of char encoder must be the same with that of word encoder
        # Initialize an LSTM Encoder object for char level
        self.__char_encoder = LSTMEncoder(self.__args.char_embedding_dim, self.__args.encoder_hidden_dim,
                                         self.__args.dropout_rate)
        # Initialize an self-attention layer for char level
        self.__char_attention = SelfAttention(self.__args.char_embedding_dim, self.__args.char_attention_hidden_dim,
                                                  self.__args.attention_output_dim, self.__args.dropout_rate)

        # Initialize an LSTM Encoder object for word level
        self.__word_encoder = LSTMEncoder(self.__args.word_embedding_dim, self.__args.encoder_hidden_dim,
                                          self.__args.dropout_rate)
        # Initialize an self-attention layer for word level
        self.__word_attention = SelfAttention(self.__args.word_embedding_dim, self.__args.word_attention_hidden_dim,
                                              self.__args.attention_output_dim, self.__args.dropout_rate)

        self.__encoder_output_dim = self.__args.encoder_hidden_dim + self.__args.attention_output_dim

        # dropout layer
        self.__dropout_layer = nn.Dropout(self.__args.dropout_rate)

        # MLP Attention
        # TODO: Insert a linear layer between encoder and MLP Attention Layer ?
        self.__char_sent_attention = MLPAttention(self.__encoder_output_dim, self.__args.dropout_rate)
        self.__word_sent_attention = MLPAttention(self.__encoder_output_dim, self.__args.dropout_rate)


        self.__intent_fusion_layer1 = FG_fusion(self.__encoder_output_dim,out_size=self.__args.slot_decoder_hidden_dim, hidden_size=self.__args.slot_decoder_hidden_dim*2,dropout_rate=self.__args.dropout_rate)

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(self.__num_intent, self.__args.intent_embedding_dim)
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent,self.__args.intent_embedding_dim)


        self.__word_slot_encoder = LSTMEncoder(self.__encoder_output_dim, self.__args.slot_decoder_hidden_dim,
                                               self.__args.dropout_rate,
                                               bidirectional=False,
                                               extra_dim=None)

        self.__char_slot_decoder = LSTMEncoder(self.__encoder_output_dim, self.__args.slot_decoder_hidden_dim,
                                               self.__args.dropout_rate,
                                               bidirectional=False,
                                               extra_dim=None)


        self.__slot_word_char_fusion_layer = FG_fusion(self.__args.slot_decoder_hidden_dim, out_size=self.__args.slot_decoder_hidden_dim, hidden_size=self.__args.slot_decoder_hidden_dim*2,dropout_rate=self.__args.dropout_rate)

        self.__slot_fusion_rate = nn.Parameter(torch.randn(1), requires_grad=True)
        self.__slot_linear_layer = nn.Linear(self.__args.slot_decoder_hidden_dim, self.__num_slot)

        self.__graph = GAT(
            nfeat=self.__args.slot_decoder_hidden_dim,nlayers=2,nheads=16,nhid=16,
            dropout=0.5,alpha=0.2,nclass=self.__args.slot_decoder_hidden_dim
            )
        self.__linear_slot_layer = nn.Linear(
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot
        )
        self.__linear_intent_layer = nn.Linear(
            self.__args.slot_decoder_hidden_dim,
            self.__num_intent
        )
    def show_summary(self):
        """
        print the abstract of the defined model.
        """
        print('Model parameters are listed as follows:\n')

        print('\tdropout rate:						                    {};'.format(self.__args.dropout_rate))
        print('\tnumber of char:						                {};'.format(self.__num_char))
        print('\tnumber of word:                                        {};'.format(self.__num_word))
        print('\tnumber of slot:                                        {};'.format(self.__num_slot))
        print('\tnumber of intent:						                {};'.format(self.__num_intent))
        print('\tchar embedding dimension:				                {};'.format(self.__args.char_embedding_dim))
        print('\tword embedding dimension:				                {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				                {};'.format(self.__args.encoder_hidden_dim))
        print('\thidden dimension of char-level self-attention:         {};'.format(self.__args.char_attention_hidden_dim))
        print('\thidden dimension of word-level self-attention:         {};'.format(self.__args.word_attention_hidden_dim))
        print('\toutput dimension of self-attention:                    {};'.format(self.__args.attention_output_dim))


        print('\tdimension of slot embedding:			                {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of intent embedding:			                {};'.format(self.__args.intent_embedding_dim))

        print('\tdimension of slot decoder hidden:  	                {};'.format(self.__args.slot_decoder_hidden_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, char_text, char_seq_lens, word_text, word_seq_lens, align_info):
        """

        :param char_text: list of list of char ids
        :param char_seq_lens: list of the number of chars, e.g. [6, 7, 7]
        :param word_text: list of list of word ids
        :param word_seq_lens: list of the number of words, e.g. [4, 3, 4]
        :param align_info: list of list of the number of chars in each word, e.g. [ [1, 2, 1, 2], [2, 2, 3], [2, 1, 3, 1] ]
        :param n_predicts:
        :param forced_slot:
        :return:
        """
        #Adjacency matrix
        def generate_adj_gat(valid,device):

            adj = torch.eye(valid+1,device=device)
            for i in range(valid+1):
                if i==0:
                    adj[:,i]=1
                    adj[i,:]=1
                elif i==1:
                    adj[i,i+1]=1
                    # adj[i,valid]=1
                elif i==valid:
                    adj[i,valid-1]=1
                    # adj[i,1]=1
                else:
                    adj[i,i-1]=1
                    adj[i,i+1]=1
            return adj


        char_tensor = self.__char_embedding(char_text)
        word_tensor = self.__word_embedding(word_text)

        # Get mask
        device = word_tensor.device
        char_rmask, char_mmask = op.generate_mask(char_seq_lens, device)
        word_rmask, word_mmask = op.generate_mask(word_seq_lens, device)


        # TODO: take masking self-attention into account
        # Pass char encoder
        char_lstm_hiddens = self.__char_encoder(char_tensor, char_seq_lens) #[batch,seq,hidden]
        char_attention_hiddens = self.__char_attention(char_tensor, mmask=char_mmask)#[batch,seq,hidden]
        char_hiddens = torch.cat([char_attention_hiddens, char_lstm_hiddens], dim=-1)#[batch,seq,hidden1+hidden2]
        char_sent_output = self.__char_sent_attention(char_hiddens, rmask=char_rmask) #[batch,dim]

        # Pass word encoder
        word_lstm_hiddens = self.__word_encoder(word_tensor, word_seq_lens)
        word_attention_hiddens = self.__word_attention(word_tensor,  mmask=word_mmask)
        word_hiddens = torch.cat([word_attention_hiddens, word_lstm_hiddens], dim=-1) #[batch,seq,hidden]

        # MLP Attention for Intent Detection
        word_sent_output = self.__word_sent_attention(word_hiddens, rmask=word_rmask)#[batch,dim]

        # Intent Prediction
        pred_intent = self.__intent_fusion_layer1(
                                                  char_sent_output,word_sent_output
                                                 ) #[batch,hidden]

        word_slot_out = self.__word_slot_encoder(word_hiddens, word_seq_lens, extra_input=None)  # [batch,seq,hid]
        # word_seq_lens:[batch,valid_len] len=seq

        flat_word_slot_out = torch.cat([word_slot_out[i][:word_seq_lens[i]]
                                        for i in range(0, len(word_seq_lens))], dim=0)  # [batch*valid_len,hid]
        aligned_word_slot_out = op.char_word_alignment(flat_word_slot_out, char_seq_lens, word_seq_lens,
                                                       align_info)  # ??


        char_slot_out = self.__char_slot_decoder(char_hiddens, char_seq_lens,
                                                 extra_input=None)
        flat_char_hiddens = torch.cat([char_slot_out[i][:char_seq_lens[i], :]
                                       for i in range(0, len(char_seq_lens))], dim=0)  # [batch*valid_len,hid]

        char_slot_out_fusion = self.__slot_word_char_fusion_layer(flat_char_hiddens, aligned_word_slot_out)

        sent_start_pos = 0
        list_intent_out, list_slot_out = [], []
        for sent_i in range(len(char_seq_lens)):
            sent_end_pos = sent_start_pos + char_seq_lens[sent_i]
            slot_hidden = char_slot_out_fusion[sent_start_pos:sent_end_pos, :]  # [valid_len,hidden]
            y = pred_intent[sent_i].unsqueeze(0)
            input_graph = torch.cat((y, slot_hidden), dim=0)  # [valid_len+1,hidden]
            adj = generate_adj_gat(char_seq_lens[sent_i], device=pred_intent.device)
            graph_out = self.__graph(input_graph.unsqueeze(0), adj).squeeze(0)
            intent_out = self.__linear_intent_layer(graph_out[0, :])
            slot_out = self.__linear_slot_layer(graph_out[1:, :])
            list_intent_out.append(intent_out)
            list_slot_out.append(slot_out)
            sent_start_pos += char_seq_lens[sent_i]

        pred_slot = torch.cat(list_slot_out, dim=0)
        pred_intent = torch.cat(list_intent_out, dim=0).reshape(pred_intent.shape[0], -1)

        return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)





class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)])
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module('attention_{}_{}'.format(i + 1, j),
                                    GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.__out_linear = nn.Linear(nhid * nheads, nclass)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2) #multi-head
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__('attention_{}_{}'.format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        if self.nlayers ==1:
            x = F.elu(self.__out_linear(x))
        else:
            x = F.elu(self.out_att(x, adj))
        return x + input



def two2three_dim(input,valid_len):
    """
    input:tensor [batch,hidden]
    valid_len:scalar [2,3,4,5...]
    """
    hidden = input.shape[-1]
    max_len = max(valid_len)
    start_pos = 0
    end_tensor = []
    device = input.device
    for i in range(len(valid_len)):
        temp1 = input[start_pos:valid_len[i]] #[seq,hidden]
        cha_len = max_len - valid_len[i]
        if cha_len > 0:
            supply_tensor1 = torch.zeros((cha_len,hidden),device=device) #[seq,hidden]
            supply_tensor2 = torch.cat([temp1,supply_tensor1],dim=0).unsqueeze(0) #[1,seq,hidden]
        else:
            supply_tensor2 = temp1.unsqueeze(0)
        end_tensor.append(supply_tensor2)
    return torch.cat(end_tensor,dim=0)
