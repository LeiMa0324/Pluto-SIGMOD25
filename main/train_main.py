import os
import sys

sys.path.append("../")

import yaml
import argparse
import torch
from bert_pytorch.trainer.phase_checker import PhaseChecker

from bert_pytorch.train_log import Trainer
import random
import numpy as np

# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    import torch
    import sys


    print("********************************* Train ********************************************")

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
    parser.add_argument("--node", type=str,default='gpu')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        options = yaml.safe_load(f)

    options['node'] = args.node

    print(options['node'])
    options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # any method should use the same data, but model dir should be method-wise
    options["data_dir"] = "../output/" + options["data_set"] + "/datasets/" + options["if_clean"] + "/"
    if options["if_clean"] == "noisy_0":
        options["anomaly_ratio"] = 0
    elif "." in options["if_clean"].split("_")[-1]:
        options["anomaly_ratio"] = float(options["if_clean"].split("_")[-1])
    else:
        options["anomaly_ratio"] = int(options["if_clean"].split("_")[-1])

    # options["anomaly_ratio"] = 0 if options["if_clean"]=="noisy_0" else int(options["if_clean"].split("_")[-1])
    options["train_path"] =  options["data_dir"]+"train"
    options["vocab_path"] = options["data_dir"]+"vocab.pkl"
    # common test data for all different noisy rates
    options["test_data_dir"] = options["data_dir"]

    run_name = options["robust_method"]

    # wandb.init(project=f"robust-anomaly_{options['data_set']}",
    #            config= options, group=options["if_clean"] )
    # wandb_name = wandb.run.name
    # wandb.run.name = run_name+"_"+wandb_name
    # wandb.run.save()
    # print(f"run file: {args.config}")

    # each method has its own model dir
    options["model_dir"] = options["data_dir"]+ run_name +"/"
    options["model_path"] = options["model_dir"] + "best_bert.pth"


    if not os.path.exists(options['model_dir']):
        os.makedirs(options['model_dir'], exist_ok=True)

    # pahse checker

    if os.path.exists(options["data_dir"] + "warm_up_model/warm_up_bert.pth"):
        warmup_model_dir = options["data_dir"] + "warm_up_model/warm_up_bert.pth"
    else:
        if os.path.exists(options["data_dir"] + "warm_up_model/best_bert.pth"):
            warmup_model_dir = options["data_dir"] + "warm_up_model/best_bert.pth"
        else:
            warmup_model_dir = None

    use_warmup_model = options["use_warmup_model"] if "use_warmup_model" in options.keys() else False
    warmup_epochs = options["warmup_epochs"] if "warmup_epochs" in options.keys() else 0
    selection_epochs = options["selection_epochs"] if "selection_epochs" in options.keys() else 0
    selection_gap = options["selection_gap"] if "selection_gap" in options.keys() else 1
    retrain_epochs = options["retrain_epochs"] if "retrain_epochs" in options.keys() else 0
    first_freeze_epochs = options["first_freeze_epochs"] if "first_freeze_epochs" in options.keys() else 0


    phase_checker = PhaseChecker(method=options["robust_method"], warmup_model_dir=warmup_model_dir,
                                 warmup_epochs=warmup_epochs,
                                 selection_epochs=selection_epochs,
                                 selection_gap=selection_gap,
                                 retrain_epochs=retrain_epochs,
                                 first_freeze_epochs=first_freeze_epochs
                                 )

    seed_everything(seed=options["seed"])
    Trainer(options, phase_checker).train()




