import sys

sys.path.append("../")
import wandb
# sys.path.append("../../")
import yaml

import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bert_pytorch.predict_logV2 import PredictorV2
from bert_pytorch.predict_logV3 import PredictorV3
from data_process.logdeep.tools.utils import *

if __name__ == "__main__":
    print("********************************* Predict ********************************************")

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
    parser.add_argument("--runname", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        options = yaml.safe_load(f)

    run_name = args.runname
    # run_name = "pluto_ITLM_fix_cluster_wandering-brook-173"
    options["output_dir"] = "../output/" + options["data_set"] + "/datasets/" + options["if_clean"] + "/"

    options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    options["data_dir"] = options["output_dir"]
    options["test_data_dir"] = "../output/" + options["data_set"] + "/datasets/" + options["if_clean"] + "/"
    # options["test_data_dir"] = "../output/" + options["data_set"] + "/datasets/"
    options["train_path"] = options["data_dir"] + "train"
    options["model_dir"] = options["output_dir"]+ run_name+"/"
    if "coteaching" in run_name:
        options["model_path"] = options["model_dir"] + "best_bert.pthmodel1"
    else:
        options["model_path"] = options["model_dir"] + "best_bert.pth"
    options["vocab_path"] = options["data_dir"] + "vocab.pkl"


    print(f"run file: {args.config}")

    if not os.path.exists(options['model_dir']):
        raise Exception(f"{options['model_dir']} does not exist!")

    PredictorV3(options).predict()







