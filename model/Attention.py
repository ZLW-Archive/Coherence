import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

# Modules
class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass

class ScaledDotProductAttention(nn.Module):

    def __init__(self, model_dim, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_constant = np.power(model_dim, 0.5)
        self.dropout_layer = nn.Dropout(attention_dropout)
        self.softmax_layer = BottleSoftmax()

    def forward(self, q, k, v, mask=None):
        attention_matrix = torch.mm(q, k.transpose(0, 1)) / self.scale_constant

        if mask is not None:
            attention_matrix.data.masked_fill_(mask, -float('inf'))

        attention_matrix = self.softmax_layer(attention_matrix)
        attention_matrix = self.dropout_layer(attention_matrix)

        output = torch.mm(attention_matrix, v)

        return output

class ScaledDotProductAttentionBatch(nn.Module):
    def __init__(self, model_dim, attention_dropout=0.1):
        super(ScaledDotProductAttentionBatch, self).__init__()
        self.scale_constant = np.power(model_dim, 0.5)
        self.dropout_layer = nn.Dropout(attention_dropout)
        self.softmax_layer = BottleSoftmax()

    def forward(self, q, k, v, mask=None):
        attention_matrix = torch.bmm(q, k.transpose(1, 2)) / self.scale_constant

        if mask is not None:
            attention_matrix.data.masked_fill_(mask, -float('inf'))

        attention_matrix = self.softmax_layer(attention_matrix)
        attention_matrix = self.dropout_layer(attention_matrix)

        output = torch.bmm(attention_matrix, v)

        return output
