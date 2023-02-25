# -*- coding: utf-8 -*-



import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.components import operation as op

MASK_VALUE = -2 ** 32 + 1


class NullOp(nn.Module):
    def forward(self, input):
        return input


class FG_fusion(nn.Module):
    def __init__(self,x_size,out_size,hidden_size,dropout_rate):
        super(FG_fusion, self).__init__()
        self.__tanh = nn.Tanh()
        self.__Wdi = nn.Linear(x_size,hidden_size)
        self.__Wai = nn.Linear(x_size,hidden_size)
        self.__Wgi = nn.Linear(2*hidden_size,1)

        self.__bilinear = nn.Bilinear(hidden_size, hidden_size, 1)


        self.__sig = nn.Sigmoid()
        self.__Wvi1 = nn.Linear(hidden_size,hidden_size)
        self.__Wvi2 = nn.Linear(hidden_size,hidden_size)

        self.__dropout = nn.Dropout(dropout_rate)

        self.__Wfi_ai1 = nn.Linear(2*hidden_size,hidden_size)
        self.__Wfi_ai2 = nn.Linear(2*hidden_size,hidden_size)

        self.__Wfi_ai = nn.Linear(2 * hidden_size, 1)

        self.__Wli = nn.Linear(hidden_size,out_size)

        # self.__W = nn.Linear(x_size,out_size)
    def forward(self,a,d):
        """
        Args:
            x:[batch,hidden]
            y: [batch,hidden]

        Returns:[batch,out_size]
        """
        di = self.__tanh(self.__Wdi(d)) #[b,h]
        ai = self.__tanh(self.__Wai(a)) #[b,h]

        gi = self.__sig(self.__bilinear(ai,di))

        vi = gi*ai + (1-gi)*di #[batch,h]

        fi1_ = self.__Wfi_ai1(torch.cat((ai,vi),dim=-1)) #[batch,h]
        fi2_=self.__Wfi_ai2(torch.cat((di,vi), dim=-1))
        fi = self.__sig(self.__Wfi_ai(torch.cat((fi1_,fi2_),dim=-1))) #[b,1]
        x = self.__dropout(vi)
        li = fi*self.__tanh(self.__Wli(x)) #[batch,out_size]

        return li


# TODO: Related to Encoder and Decoder

class EmbeddingCollection(nn.Module):
    """
    TODO: Provide position vector encoding
    Provide word vector encoding.
    """
    def __init__(self, input_dim, embedding_dim, max_len=5000):
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        # Here embedding_dim must be an even embedding.
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        # Word vector encoder.
        self.__embedding_layer = nn.Embedding(
            self.__input_dim, self.__embedding_dim
        )

    def forward(self, input_x):
        # Get word vector encoding.
        embedding_x = self.__embedding_layer(input_x)

        # Board-casting principle.
        return embedding_x


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate, bidirectional=True, extra_dim=None):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.__dropout_rate = dropout_rate
        self.__bidirectional = bidirectional
        self.__extra_dim = extra_dim

        lstm_input_dim = self.__embedding_dim + (0 if self.__extra_dim is None else self.__extra_dim)

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.__hidden_dim, batch_first=True,
                                    bidirectional=self.__bidirectional, dropout=self.__dropout_rate, num_layers=1)

    def forward(self, embedded_text, seq_lens, extra_input=None):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([embedded_text, extra_input], dim=-1)
        else:
            input_tensor = embedded_text

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(input_tensor)


        padded_hiddens, _ = op.pack_and_pad_sequences_for_rnn(dropout_text,
                                                              torch.tensor(seq_lens, device=dropout_text.device),
                                                              self.__lstm_layer)

        # return torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)
        return padded_hiddens



class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate, input_linear=True, bilinear=False):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__input_linear = input_linear
        self.__bilinear = bilinear

        # Declare network structures.
        if input_linear and not bilinear:
            self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
            self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
            self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        elif bilinear:
            self.__linear = nn.Linear(self.__query_dim, self.__key_dim)

        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value, mmask=None):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query) if self.__input_linear and not self.__bilinear else input_query
        linear_key = self.__key_layer(input_key) if self.__input_linear and not self.__bilinear else input_key
        linear_value = self.__value_layer(input_value) if self.__input_linear and not self.__bilinear else input_value

        if self.__input_linear and not self.__bilinear:
            score_tensor = torch.matmul(linear_query, linear_key.transpose(-2, -1)) / math.sqrt(
                self.__hidden_dim if self.__input_linear else self.__query_dim)
        elif self.__bilinear:
            score_tensor = torch.matmul(self.__linear(linear_query), linear_key.transpose(-2, -1))

        if mmask is not None:
            assert mmask.shape == score_tensor.shape
            score_tensor = mmask * score_tensor + (1 - mmask) * MASK_VALUE

        score_tensor = F.softmax(score_tensor, dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        # forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, mmask=None):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(dropout_x, dropout_x, dropout_x, mmask=mmask)

        return attention_x



class AttentiveModule(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, output_dim, dropout_rate, bilinear=False):
        super(AttentiveModule, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__attention_layer = QKVAttention(
            self.__query_dim, self.__key_dim, self.__value_dim, 0, 0, self.__dropout_rate,
            input_linear=False, bilinear=bilinear
        )

        self.__ffn = FFN(self.__value_dim, self.__output_dim, self.__output_dim)

    def forward(self, input_query, input_key, input_value, mmask=None):
        """

        :param input_query:
        :param input_key:
        :param input_value:
        :param mmask:
        :return:
        """
        att = self.__attention_layer(input_query, input_key, input_value, mmask=mmask)

        z = self.__ffn(att)

        return z


class MLPAttention(nn.Module):

    def __init__(self, input_dim, dropout_rate):
        super(MLPAttention, self).__init__()

        # Record parameters
        self.__input_dim = input_dim
        self.__dropout_rate = dropout_rate

        # Define network structures
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__sent_attention = nn.Linear(self.__input_dim, 1, bias=False)

    def forward(self, encoded_hiddens, rmask=None):
        """
        Merge a sequence of word representations as a sentence representation.
        :param encoded_hiddens: a tensor with shape of [bs, max_len, dim]
        :param rmask: a mask tensor with shape of [bs, max_len]
        :return:
        """
        # TODO: Do dropout ?
        dropout_input = self.__dropout_layer(encoded_hiddens)
        score_tensor = self.__sent_attention(dropout_input).squeeze(-1)

        if rmask is not None:
            assert score_tensor.shape == rmask.shape, "{} vs {}".format(score_tensor.shape, rmask.shape)
            score_tensor = rmask * score_tensor + (1 - rmask) * MASK_VALUE

        score_tensor = F.softmax(score_tensor, dim=-1)
        # matrix multiplication: [bs, 1, max_len] * [bs, max_len, dim] => [bs, 1, dim]
        sent_output = torch.matmul(score_tensor.unsqueeze(1), dropout_input).squeeze(1)

        return sent_output #[batch,dim]


# TODO: Related GNN layers

class MultiHeadAtt(nn.Module):
    def __init__(self, nhid, keyhid, nhead=10, head_dim=10, dropout=0.1, if_g=False):
        super(MultiHeadAtt, self).__init__()

        if if_g:
            self.WQ = nn.Conv2d(nhid * 3, nhead * head_dim, 1)
        else:
            self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(keyhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(keyhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(nhid)

        self.nhid, self.nhead, self.head_dim = nhid, nhead, head_dim

    def forward(self, query_h, value, mask, query_g=None):

        if not (query_g is None):
            query = torch.cat([query_h, query_g], -1)
        else:
            query = query_h
        query = query.permute(0, 2, 1)[:, :, :, None]
        value = value.permute(0, 3, 1, 2)

        residual = query_h
        nhid, nhead, head_dim = self.nhid, self.nhead, self.head_dim

        B, QL, H = query_h.shape

        _, _, VL, VD = value.shape  # VD = 1 or VD = QL

        assert VD == 1 or VD == QL
        # q: (B, H, QL, 1)
        # v: (B, H, VL, VD)
        q, k, v = self.WQ(query), self.WK(value), self.WV(value)

        q = q.view(B, nhead, head_dim, 1, QL)
        k = k.view(B, nhead, head_dim, VL, VD)
        v = v.view(B, nhead, head_dim, VL, VD)

        alpha = (q * k).sum(2, keepdim=True) / np.sqrt(head_dim)
        alpha = alpha.masked_fill(mask[:, None, None, :, :], -np.inf)
        alpha = self.drop(F.softmax(alpha, 3))
        att = (alpha * v).sum(3).view(B, nhead * head_dim, QL, 1)

        output = F.leaky_relu(self.WO(att)).permute(0, 2, 3, 1).view(B, QL, H)
        output = self.norm(output + residual)

        return output




#
# #激活函数
# def squash(inputs,axis=-1):
#     """
#     inputs:
#     :param inputs:
#     :param axis:
#     :return:
#     """
#     norm = torch.norm(inputs,p=2,dim=axis,keepdim=True)
#     scale = norm**2/(1+norm**2)/(norm+1e-8)
#     return scale*inputs
#
# #将数据维度变为胶囊可以接受的维度
# class PrimaryCapsule(nn.Module):
#     """
#     :return: output tensor, size=[batch, num_caps, dim_caps]
#     """
#     def __init__(self, in_dim, out_num_caps,dim_caps):
#         super(PrimaryCapsule, self).__init__()
#         self.dim_caps = dim_caps
#         self.in_dim = in_dim
#         self.linear = nn.Linear(in_dim,out_num_caps*dim_caps)
#         self.dropout = nn.Dropout(0.2)
#     def forward(self, x):
#         """
#         :param x: [batch,dim]
#         :return:[batch,caps_num,dim_caps]
#         """
#         outputs = self.dropout(self.linear(x)) #[batch,caps_num*dim_caps]
#         outputs = outputs.view(x.size(0), -1, self.dim_caps)
#         return squash(outputs)
#
# class DenseCapsule(nn.Module):
#     def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
#         super(DenseCapsule, self).__init__()
#         self.in_num_caps = in_num_caps
#         self.in_dim_caps = in_dim_caps
#         self.out_num_caps = out_num_caps
#         self.out_dim_caps = out_dim_caps
#         self.routings = routings
#         self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
#
#     def forward(self, x):
#         """
#         :param x: [batch, in_num_caps, in_dim_caps]
#         :return:
#         """                                             #[batch,1,in_num_caps,in_dim_caps,1]
#         x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1) #[batch,out_num_caps,in_num_caps,out_dim_caps]
#         x_hat_detached = x_hat.detach() #截断梯度反向传播
#
#         # [batch,out_num_caps,in_num_caps]
#         b = torch.zeros((x.size(0), self.out_num_caps, self.in_num_caps),device=x.device)
#         #路由机制
#         assert self.routings > 0, 'The \'routings\' should be > 0.'
#         for i in range(self.routings):
#             c = F.softmax(b, dim=1) #[batch,out_num_caps,in_num_caps]
#             if i == self.routings - 1:
#                 outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
#             else:
#                 outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))#[batch,out_num_caps,1,out_dim_caps]
#                 b = b + torch.sum(outputs * x_hat_detached, dim=-1)
#         return torch.squeeze(outputs, dim=-2) #[batch,out_num_caps,out_dim_caps]
#
# class CapsuleNet(nn.Module):
#     """
#     A
#     :param routings: number of routing iterations
#     Shape:
#         - Input: (batch, channels, width, height), optional (batch, classes) .
#         - Output:((batch, classes), (batch, channels, width, height))
#     """
#     def __init__(self, input_size,in_num_caps,in_dim_caps, classes, routings):
#         super(CapsuleNet, self).__init__()
#         self.input_size = input_size
#         self.classes = classes
#         self.routings = routings
#
#         # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
#         self.primarycaps_x = PrimaryCapsule(input_size, in_num_caps,in_dim_caps)
#         self.primarycaps_y = PrimaryCapsule(input_size, in_num_caps,in_dim_caps)
#
#         # Layer 3: Capsule layer. Routing algorithm works here.
#         self.digitcaps_x = DenseCapsule(in_num_caps=in_num_caps, in_dim_caps=in_dim_caps,
#                                       out_num_caps=classes, out_dim_caps=classes, routings=routings)
#         self.digitcaps_y = DenseCapsule(in_num_caps=in_num_caps, in_dim_caps=in_dim_caps,
#                                         out_num_caps=classes, out_dim_caps=classes, routings=routings)
#         # Decoder network.
#         self.decoder_x = nn.Sequential(
#             nn.Linear(classes, classes*2),
#             nn.ReLU(inplace=True),
#             nn.Linear(classes*2, classes*2),
#             nn.ReLU(inplace=True),
#             nn.Linear(classes*2, classes),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(classes*2, classes * 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(classes * 2, classes * 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(classes * 2, classes),
#         )
#
#
#     def forward(self, x,y):
#         """
#         Args:
#             x: [batch,dim]
#             y:
#         Returns:
#         """
#         x = self.primarycaps_x(x) #[batch,in_num_caps,in_dim_caps]
#         x = self.digitcaps_x(x) #[batch,out_num_caps,out_dim_caps]
#         length_x = x.norm(dim=-1) #[batch,out_num_caps]
#         # output_x = self.decoder(length_x) #[batch,intent_num]
#
#         y = self.primarycaps_y(y)  # [batch,in_num_caps,in_dim_caps]
#         y = self.digitcaps_y(y)  # [batch,out_num_caps,out_dim_caps]
#         length_y = y.norm(dim=-1)  # [batch,out_num_caps]
#         # output_y = self.decoder(length_y)  # [batch,intent_num]
#         output = torch.cat((length_x,length_y),dim=-1)
#
#         output = self.decoder(output)
#
#         return output