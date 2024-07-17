import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward, LayerNorm


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, output_attentions=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.output_attentions = output_attentions
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, output_attentions=self.output_attentions)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.norm = LayerNorm(size=hidden)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, attention_weights = None):
        # multi-head attention layer
        hidden_states =  self.norm(x)
        attention_outputs = self.attention(hidden_states, hidden_states, hidden_states,mask=mask, attention_weights= attention_weights)
        hidden_states = attention_outputs[0]
        hidden_states = self.dropout(hidden_states)+x
        # feed-forward layers
        hidden_states = self.dropout(self.output_sublayer(hidden_states, self.feed_forward))
        outputs = (hidden_states,)
        if self.output_attentions:
            attention_score = attention_outputs[-1]
            outputs= outputs + (attention_score,)

        return outputs
