import numpy as np
# import wandb
from torch.utils.data import DataLoader
from bert_pytorch.model import BERT
from bert_pytorch.trainer.phase_checker import PhaseChecker
from bert_pytorch.trainer.BertTrainer_bk import BERTTrainerV2
from bert_pytorch.trainer.BertTrainer_coteaching import BERTTrainer_coteaching
from bert_pytorch.dataset import LogDataset, WordVocab
from bert_pytorch.dataset.sample import generate_train_valid
from bert_pytorch.dataset.utils import save_parameters
from bert_pytorch.trainer.sample_selector import Sample_Selector
from torch.autograd import Variable

from bert_pytorch.predict_logV2 import PredictorV2
from bert_pytorch.dataset.utils import bert_read_file
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import tqdm
import gc
import ast
import os



class Trainer():
    def __init__(self, options, phase_checker):
        self.device = options["device"]
        self.data_dir = options["data_dir"]

        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.train_path = options["train_path"]

        self.seq_len = options["seq_len"]
        self.max_len = options["max_len"]

        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.lr = options["lr"]
        self.adam_beta1 = options["adam_beta1"]
        self.adam_beta2 = options["adam_beta2"]
        self.adam_weight_decay = options["adam_weight_decay"]
        self.with_cuda = options["with_cuda"]
        self.cuda_devices = options["cuda_devices"]
        self.log_freq = options["log_freq"]

        self.hidden = options["hidden"]
        self.layers = options["layers"]
        self.attn_heads = options["attn_heads"]

        self.n_epochs_stop = options["n_epochs_stop"]

        self.mask_ratio = options["mask_ratio"]
        self.is_label = options["is_label"]

        self.robust_method = options["robust_method"]
        self.if_hypersphere_loss = self.robust_method == 'logbert'

        self.anomaly_ratio = options["anomaly_ratio"]
        self.num_candidates_ratio = options["num_candidates_ratio"]

        self.store_embedding = options["store_embedding"] if "store_embedding" in options.keys() else False

        self.phase_checker = phase_checker
        self.epochs = self.phase_checker.epochs # automatically get max epochs from phase checker


        self.if_predict = options["if_predict"] if "if_predict" in options.keys() else False
        self.p_thres = options["p_thres"] if "p_thres" in options.keys() else 0.5
        self.seq_thres = options["seq_thres"] if "seq_thres" in options.keys() else 0.1
        self.outlier_cluster_thres = options["outlier_cluster_thres"] if "outlier_cluster_thres" in options.keys() else 80
        self.cluster_num = options["cluster_num"] if "cluster_num" in options.keys() else 20
        self.ITLM_percent = options["ITLM_percent"] if "ITLM_percent" in options.keys() else 0.6

        self.if_process_outlier = options["if_process_outlier"] if "if_process_outlier" in options.keys() else True
        self.predictor = PredictorV2(options)

        print("Save options parameters")
        save_parameters(options, self.model_dir + "parameters.txt")


    def train(self):

        print("Loading vocab", self.vocab_path)
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        print("vocab Size: ", len(self.vocab))

        print("\nLoading Train Dataset")

        train_file_path = self.train_path+"train.csv"

        seq_id_train, logkey_train, label_train, token_label_train= bert_read_file(train_file_path)
        print(f"\nLoading Training Dataset from {train_file_path}")


        # mask the log keys, the labels of the unmasked keys are 0, which is ignored during loss computation
        train_dataset = LogDataset(logkey_train.tolist(), self.vocab, seq_len=self.seq_len,
                                    mask_ratio=self.mask_ratio,
                                   label_corpus=label_train.tolist(), token_label_corpus=token_label_train.tolist(), seq_id = seq_id_train.tolist())


        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                      collate_fn=train_dataset.collate_fn, drop_last=True, shuffle=True)
        # self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        #                                collate_fn=train_dataset.collate_fn, drop_last=True)


        print("Building BERT model")

        bert = BERT(len(self.vocab), max_len=self.max_len, hidden=self.hidden, n_layers=self.layers, attn_heads=self.attn_heads,
                     output_attentions= False)

        print("Creating BERT Trainer")
        if 'coteaching' in self.robust_method:
            self.trainer = BERTTrainer_coteaching(bert, len(self.vocab), anomaly_ratio= self.anomaly_ratio, model_dir =self.model_dir, train_dataloader=self.train_data_loader, valid_dataloader=None,
                              lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay,
                              with_cuda=self.with_cuda,
                               robust_method=self.robust_method,  store_embedding=self.store_embedding,
                                   train_dataset= train_dataset, valid_dataset=None,
                                   phase_checker= self.phase_checker,
                                   num_candidates_ratio = self.num_candidates_ratio
                                   )
        else:
            self.sample_selector = Sample_Selector( anomaly_ratio= self.anomaly_ratio,
                                                   train_dataset = train_dataset,
                                                    output_dir = self.model_dir,
                                                    selection_method= self.robust_method,
                                                    p_thres = self.p_thres, ITLM_percent= self.ITLM_percent,
                                                    outlier_cluster_thres = self.outlier_cluster_thres,
                                                    if_process_outlier= self.if_process_outlier,
                                                    cluster_num = self.cluster_num
                                                    )

            self.trainer = BERTTrainerV2(bert, train_file_path, self.vocab_path, len(self.vocab),  model_dir =self.model_dir, train_dataloader=self.train_data_loader, valid_dataloader=None,
                                  lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay,
                                  with_cuda=self.with_cuda,
                                   robust_method=self.robust_method,  store_embedding=self.store_embedding,
                                       train_dataset= train_dataset, valid_dataset=None,
                                       phase_checker= self.phase_checker,
                                       num_candidates_ratio = self.num_candidates_ratio,
                                        sample_selector = self.sample_selector, seq_thres= self.seq_thres
                                       )
        del train_dataset
        del logkey_train
        gc.collect()

        self.start_iteration(surfix_log="log2")
        memory = torch.cuda.max_memory_allocated(device=None)
        print(f"max memory allocated: {round(memory / (1024 ** 3), 2)}")

        # self.plot_detailed_loss()

#only train with minimizing the hypersphere size
    def start_iteration(self, surfix_log):
        print("Training Start")

        for epoch in range(self.epochs):
            print("\n")

            if self.if_hypersphere_loss:
                center = self.calculate_center([self.train_data_loader])
                self.trainer.hyper_center = center

            avg_train_loss, train_dist = self.trainer.train(epoch)   # train with masked language model, return avglost, distance

            if self.if_hypersphere_loss:
                self.trainer.radius = self.trainer.get_radius(train_dist , self.trainer.nu)

            self.trainer.save_log(self.model_dir, surfix_log)
            self.save_model()

            train_metrics = {"train/epoch": self.trainer.log["train"]["epoch"][-1],
                "train/lr": self.trainer.log["train"]["lr"][-1],
                "train/train_loss": self.trainer.log["train"]["loss"][-1]}

            # wandb.log(train_metrics, step=epoch)

            if self.phase_checker.if_predict(epoch) and self.if_predict:
                if 'coteaching' in self.robust_method:
                    self.predictor.predict(epoch, self.trainer.model1)  # predict model 1

                    pred_metrics = {"pred/epoch": self.predictor.log["epoch"][-1],
                                    "pred/precision_model1": self.predictor.log["precision"][-1],
                                    "pred/recall_model1": self.predictor.log["recall"][-1],
                                    "pred/f-1_model1": self.predictor.log["f-1"][-1],
                                    "pred/TP_model1": self.predictor.log["TP"][-1],
                                    "pred/TN_model1": self.predictor.log["TN"][-1],
                                    "pred/FP_model1": self.predictor.log["FP"][-1],
                                    "pred/FN_model1": self.predictor.log["FN"][-1],
                                    "pred/auc_model_1": self.predictor.log["auc"][-1]}
                    # wandb.log(pred_metrics, step=epoch)

                    self.predictor.predict(epoch, self.trainer.model2)  # predict model 2

                    pred_metrics = {"pred/epoch": self.predictor.log["epoch"][-1],
                                    "pred/precision_model2": self.predictor.log["precision"][-1],
                                    "pred/recall_model2": self.predictor.log["recall"][-1],
                                    "pred/f-1_model2": self.predictor.log["f-1"][-1],
                                    "pred/TP_model2": self.predictor.log["TP"][-1],
                                    "pred/TN_model2": self.predictor.log["TN"][-1],
                                    "pred/FP_model2": self.predictor.log["FP"][-1],
                                    "pred/FN_model2": self.predictor.log["FN"][-1],
                                    "pred/auc_model_2": self.predictor.log["auc"][-1]}
                    # wandb.log(pred_metrics, step=epoch)
                else:
                    # update the vocab of predictor
                    self.predictor.vocab = self.trainer.vocab
                    self.predictor.predict(epoch, self.trainer.model)  #predict
                    # todo: multiple test data to predict

                    pred_metrics = {"pred/epoch": self.predictor.log["epoch"][-1],
                                    "pred/precision": self.predictor.log["precision"][-1],
                                    "pred/recall": self.predictor.log["recall"][-1],
                                    "pred/f-1": self.predictor.log["f-1"][-1],
                                    "pred/TP": self.predictor.log["TP"][-1],
                                    "pred/TN": self.predictor.log["TN"][-1],
                                    "pred/FP": self.predictor.log["FP"][-1],
                                    "pred/FN": self.predictor.log["FN"][-1],
                                    "pred/auc": self.predictor.log["auc"][-1]}
                    # wandb.log(pred_metrics, step=epoch)

        # wandb.finish()

    def save_model(self):
        self.trainer.save(self.model_path)

    def calculate_center(self, data_loader_list):
        print("start calculate center")
        # model = torch.load(self.model_path)
        # model.to(self.device)
        with torch.no_grad():
            outputs = 0
            total_samples = 0
            for data_loader in data_loader_list:
                totol_length = len(data_loader)
                data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    result = self.trainer.model.forward(data["bert_input"])
                    cls_output = result["cls_output"]

                    outputs += torch.sum(cls_output.detach().clone(), dim=0)
                    total_samples += cls_output.size(0)

        center = outputs / total_samples

        return center
