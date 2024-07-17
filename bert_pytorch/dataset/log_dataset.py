import numpy
from torch.utils.data import Dataset
import torch
import random
import numpy as np
from collections import defaultdict

class LogDataset(Dataset):
    def __init__(self, log_corpus,  vocab, seq_len,  encoding="utf-8", predict_mode=False, mask_ratio=0.15,
                 label_corpus=None, token_label_corpus=None, seq_id=None) -> object:
        """

        :rtype: object
        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.encoding = encoding

        self.predict_mode = predict_mode
        self.log_corpus = log_corpus #语料库

        self.label_corpus = label_corpus  # abnormal ground truth label
        self.token_label_corpus = token_label_corpus  # abnormal ground truth label of each token
        self.corpus_lines = len(log_corpus)
        self.seq_id = seq_id

        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):

        # k a sequence, t a sequence of timestamp, the ground truth label of abnormal for k
        seq_id, k, l, t_l =self.seq_id[idx], self.log_corpus[idx],  self.label_corpus[idx], self.token_label_corpus[idx]

        k_before_mask = list(k)
        for i, token in enumerate(k_before_mask):
            k_before_mask[i]=self.vocab.stoi.get(token, self.vocab.unk_index)

        k_before_mask = [self.vocab.pad_index] + k_before_mask

        # k the masked sequence, the token of the masked k, scores: the abnormal score of each token
        k_masked, k_label= self.random_masking(k)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        k = [self.vocab.sos_index] + k_masked  # put the CLS at first
        k_label = [self.vocab.pad_index] + k_label
        # k_label = [self.vocab.sos_index] + k_label

        token_labels = [0]+t_l


        return  k, k_label, l, token_labels, k_before_mask, idx, seq_id

    def random_masking(self, k):

        tokens = list(k)
        output_label = []


        for i, token in enumerate(tokens):

            prob = random.random()
            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError("no mask in visualization")

                if self.predict_mode:
                    tokens[i] = self.vocab.mask_index
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                    continue

                prob /= self.mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                # label of the masked token is itself
                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                # the label of the unmasked token is 0
                output_label.append(0)


        return tokens, output_label

    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]

        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = defaultdict(list)

        for seq in batch:

            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]

            label = seq[2]
            token_label = seq[3][:seq_len]
            bert_ori_input = seq[4][:seq_len]
            index = seq[5]
            seq_id = seq[6]

            padding = [self.vocab.pad_index for _ in range(seq_len - len(bert_input))]
            token_label_padding = [-1 for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), token_label.extend(token_label_padding), bert_ori_input.extend(padding)


            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["label"].append(label)
            # output["token_label"].append(token_label)
            output["bert_ori_input"].append(bert_ori_input)
            output["index"].append(index)
            output["seq_id"].append(seq_id)

        output["bert_input"] = torch.tensor(np.array(output["bert_input"]), dtype=torch.long)
        output["bert_label"] = torch.tensor(np.array(output["bert_label"]), dtype=torch.long)
        output["label"] = torch.tensor(np.array(output["label"]), dtype=torch.int)
        # token_label_arr = np.array(output["token_label"])
        # output["token_label"] = torch.from_numpy(token_label_arr)
        output["bert_ori_input"] = torch.tensor(np.array(output["bert_ori_input"]), dtype=torch.long)
        output["index"] = torch.tensor(np.array(output["index"]), dtype=torch.long)
        output["seq_id"] = torch.tensor(np.array(output["seq_id"]), dtype=torch.long)


        return output

    def get_indices_of_seq_ids(self, seq_ids):
        return np.in1d(np.array(self.seq_id), np.array(seq_ids)).nonzero()[0]
