import random
import os
import numpy as np
import torch
import pandas as pd
import ast


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))

# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def convert_list_string_to_list(string, separator = ' ', dtype=int):
    res = string.split(separator)
    res = [x.strip('[]\'\"\n') for x in res]
    res = [dtype(x) for x in res]
    return res

def bert_read_file(data_path = None, min_votes = None, data = None):
    if data is None:
        data = pd.read_csv(data_path)

    seq_ids = data["seq_id"].to_numpy()
    # np array of string list
    logkey_seqs = data["logkey_seq"].map(lambda x: [str(s) for s in ast.literal_eval(x)]).to_numpy()
    seq_anomaly_labels = data["seq_anomaly_label"].to_numpy()
    # np array of int list
    if "token_anomaly_label" in data.columns:
        token_anomaly_labels = data["token_anomaly_label"].map(lambda x: [int(tl) for tl in ast.literal_eval(x)]).to_numpy()
        # token_anomaly_labels = token_anomaly_labels.astype(np.int)
    else:
        token_anomaly_labels = data["logkey_seq"].map(lambda x: np.zeros(len(ast.literal_eval(x))).tolist()).to_numpy()
    return  seq_ids, logkey_seqs, seq_anomaly_labels, token_anomaly_labels

def deeplog_read_file(data_path):
    data = pd.read_csv(data_path)

    data["Sequentials"] = data["Sequentials"].map(lambda x: ast.literal_eval(x))
    next_token_labels = data["next_token_label"].to_numpy()

    return  data, next_token_labels