import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, output_attentions=False):
        super().__init__()
        assert d_model % h == 0
        self.output_attentions = output_attentions

        # We assume d_v always equals d_k
        self.d_k = d_model // h  #地板除， the dimension of key matrix
        self.h = h

        # initiate three linear networks for q, k, v, matrix size = d_model
        # d_model 必须为head_num的multiple
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

# forward for all the attention heads
    def forward(self, query, key, value, mask=None, attention_weights= None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  #tranpose between d_1 and d_2
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # as initiation, q,k,v are all x, pass x thru three linear networks to get initial q, k, v
        #view()方法只适用于满足连续性条件的tensor，并且该操作不会开辟新的内存空间，只是产生了对原存储空间的一个新别称和引用，返回值是视图。
        # 而reshape()方法的返回值既可以是视图，也可以是副本，当满足连续性条件时返回view，否则返回副本[ 此时等价于先调用contiguous()方法在使用view() ]


        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout, attention_weights= attention_weights)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        outputs = (self.output_linear(x),)

        if self.output_attentions:
            outputs = outputs + (attn,)

        return outputs
