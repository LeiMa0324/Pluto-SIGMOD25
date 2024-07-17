import sys

sys.path.append("../")
import wandb
import yaml


import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bert_pytorch.dataset import WordVocab

import torch


if __name__ == "__main__":
    if torch.cuda.is_available():
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"total memory {meminfo.total / 1024 / 1024} GB")
        print(f"used memory {meminfo.used / 1024 / 1024} GB")
        print(f"free memory {meminfo.free / 1024 / 1024} GB")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        options = yaml.safe_load(f)

    options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # any method should use the same data, but model dir should be method-wise
    options["data_dir"] = "../output/" + options["data_set"] + "/datasets/" + options["if_clean"] + "/"

    options["train_path"] =  options["data_dir"]+"traintrain.csv"
    options["vocab_path"] = options["data_dir"]+"vocab.pkl"
    print("********************************* Vocab ********************************************")
    print(f"run file: {args.config}")


    vocab = WordVocab(train_path=options["train_path"], if_seq_label=True, if_token_label=True)
    print("vocab_size", len(vocab))
    vocab.save_vocab(options["vocab_path"])
    print(f"vocab path : {options['vocab_path']}")





