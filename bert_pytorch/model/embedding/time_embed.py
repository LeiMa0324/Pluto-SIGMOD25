import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        # nn.Linear(in_features, out_features), the input size is [batch_size, ]
        self.time_embed = nn.Linear(1, embed_size)

    def forward(self, time_interval):
        return self.time_embed(time_interval)
