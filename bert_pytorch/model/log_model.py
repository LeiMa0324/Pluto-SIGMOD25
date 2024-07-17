import torch.nn as nn
import torch
from .bert import BERT
from .meta_module import MetaModule

class BERTLog(MetaModule):
    """
    BERT Log Model
    """

    #todo: make BERTLog inherit the MetaModule

    def __init__(self, bert: BERT, vocab_size, output_attentions = False):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLogModel(self.bert.hidden, vocab_size)
        self.time_lm = TimeLogModel(self.bert.hidden)
        # self.fnn_cls = LinearCLS(self.bert.hidden)
        #self.cls_lm = LogClassifier(self.bert.hidden)
        self.result = {"logkey_output": None, "time_output": None, "cls_output": None, "cls_fnn_output": None, "atten_scores":None}
        self.output_attentions = output_attentions


    def forward(self, x, time_info=None, attention_weights = None):
        # outputs: batch_size, seq_len, hidden_size
        outputs = self.bert(x, time_info=time_info, attention_weights= attention_weights)
        x = outputs[0]
        if self.output_attentions:
            self.result["atten_scores"] = outputs[-1]

        # results: batch_size, seq_len, vocab_size
        self.result["logkey_output"] = self.mask_lm(x)
        if time_info is not None:
            self.result["time_output"] = self.time_lm(x)

        # self.result["cls_output"] = x.float().mean(axis=1) #x[:, 0]
        self.result["cls_output"] = x[:, 0]
        # self.result["cls_output"] = self.fnn_cls(x[:, 0])

        # print(self.result["cls_fnn_output"].shape)

        return self.result

# the fully connected layer for masked log prediction after bert
class MaskedLogModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

# the fully connected layer for masked log time prediction after bert
class TimeLogModel(nn.Module):
    def __init__(self, hidden, time_size=1):
        super().__init__()
        self.linear = nn.Linear(hidden, time_size)

    def forward(self, x):
        return self.linear(x)

class LogClassifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, cls):
        return self.linear(cls)

class LinearCLS(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x)