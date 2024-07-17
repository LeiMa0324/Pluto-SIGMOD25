import torch.nn as nn
import torch.nn.functional as F
import torch


import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None, attention_weights= None):
        # score = Q*K/sqrt(Q.size)
        # query, key.size = (batch_size, h，seq_len, d_k)
        # score.size = (batch_size, h，seq_len, seq_len)
        # attention_weights.size = (batch_size, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)  # normalize the original score into range(0,1)

        if attention_weights is not None:
            #(batch_size, seq_len, seq_len)->(batch_size,1, seq_len, seq_len)->(batch_size,h, seq_len, seq_len)
            attention_weights = attention_weights.unsqueeze(1).repeat(1,scores.size()[1],1,1)
            p_attn = p_attn*attention_weights
            p_attn = F.softmax(p_attn, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # torch.matmul(p_attn, value) the attentioned context, p_attn the attention scores
        return torch.matmul(p_attn, value), p_attn
