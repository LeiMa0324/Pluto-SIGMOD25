import math
import os.path
import time

from bert_pytorch.trainer.phase_checker import PhaseChecker
import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from .optim_schedule import ScheduledOptim
from ..model import BERTLog, BERT
from .sample_selector import Sample_Selector
from bert_pytorch.dataset import LogDataset
import torch.nn.functional as F
from sample.sampler.CO_TEACHING.co_teaching import coteaching


torch.cuda.empty_cache()

# todo: create a worm-up model to save time

class BERTTrainer_coteaching:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.
    """
    def __init__(self, bert: BERT, vocab_size: int, anomaly_ratio,  model_dir,  num_candidates_ratio,
                 phase_checker: PhaseChecker,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 train_dataset: LogDataset =None, valid_dataset: LogDataset = None,
                 test_normal_dataloader: DataLoader =None, test_abnormal_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=5000,
                 with_cuda: bool = True,
                 robust_method = None,  store_embedding = False):


        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.robust_method = robust_method

        self.model_dir = model_dir
        self.num_candidates_ratio = num_candidates_ratio
        self.num_candidates = int(self.num_candidates_ratio * vocab_size)

        # Setting the train and valid data loader

        self.train_data_loader = train_dataloader
        self.valid_data_loader = valid_dataloader

        self.test_normal_data_loader = test_normal_dataloader
        self.test_abnormal_data_loader = test_abnormal_dataloader

        self.train_dataset = train_dataset # the full data set of training
        self.valid_dataset = valid_dataset # the full data set of validation


        # This BERT model will be saved every epoch

        self.vocab_size = vocab_size

        self.phase_checker = phase_checker
        # Initialize the BERT Language Model, with BERT model
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.bert1 = bert
        self.model1 = BERTLog(bert, vocab_size, output_attentions = False).to(self.device)

        self.optim1 = Adam(self.model1.parameters(), lr= self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule1 = ScheduledOptim(self.optim1, self.bert1.hidden, n_warmup_steps=self.warmup_steps, lr=self.lr)

        self.bert2 = BERT(self.bert1.vocab_size, max_len=self.bert1.max_len, hidden=self.bert1.hidden,
                          n_layers=self.bert1.n_layers, attn_heads=self.bert1.attn_heads,
                     output_attentions= False)
        self.model2 = BERTLog(self.bert2, vocab_size, output_attentions = False).to(self.device)

        self.optim2 = Adam(self.model2.parameters(), lr= self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule2 = ScheduledOptim(self.optim2, self.bert2.hidden, n_warmup_steps=self.warmup_steps, lr=self.lr)


        self.coteacher = coteaching(anomaly_percent= anomaly_ratio, epochs = phase_checker.epochs, method = robust_method)


        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # ignore the loss of label = 0
        self.criterion = nn.NLLLoss(ignore_index=0, reduce=False)  #return a scalar

        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

        print("Total Parameters:", sum([p.nelement() for p in self.model1.parameters()]))

        self.epochs = self.phase_checker.epochs

    def train(self, epoch):
        return self.coteaching_selection_iteration(epoch, self.train_data_loader, start_train=True)



    # vanilla iteration without robust method
    def coteaching_selection_iteration(self, epoch, data_loader, start_train):

        str_code = "train" if start_train else "valid"
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['time'].append(start)
        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        total_loss = 0.0  # total loss
        total_vector_loss = 0.0  # the loss of hypersphere
        total_dist = []

        lr1 = self.optim1.state_dict()['param_groups'][0]['lr']
        lr2 = self.optim2.state_dict()['param_groups'][0]['lr']
        self.log[str_code]['lr'].append(lr1)

        total_anomaly_model1 = 0
        total_model1 = 0
        total_anomaly_model2 = 0
        total_model2 = 0

        phase = self.phase_checker.get_phase(epoch)
        selection_df = pd.DataFrame()

        for i, data in data_iter:

            input_tensor = data["bert_input"].to(self.device)
            label_tensor = data["bert_label"].to(self.device)
            result1 = self.model1.forward(input_tensor)
            mask_lm_output1 = result1["logkey_output"]
            mask_loss1 = self.criterion(mask_lm_output1.transpose(1, 2), label_tensor)
            seq_loss1 = mask_loss1.sum(dim=1).detach().cpu().numpy().tolist()

            result2 = self.model2.forward(input_tensor)
            mask_lm_output2 = result2["logkey_output"]
            mask_loss2 = self.criterion(mask_lm_output2.transpose(1, 2), label_tensor)
            seq_loss2 = mask_loss2.sum(dim=1).detach().cpu().numpy().tolist()

            mask_num = (torch.sum(data["bert_label"] > 0, dim=1)).detach().cpu().numpy().tolist()


            df = pd.DataFrame()
            df["seq_label"] = data["label"].detach().cpu().numpy()
            df["seq_id"] = data["seq_id"].detach().cpu().numpy()
            df["loss_1"] = seq_loss1
            df["loss_2"] = seq_loss2
            df["mask_num"] = mask_num


            anomaly_model_1, anomaly_model_2,num_model1, num_model2,  model1_sel_tensor, model2_sel_tensor, df = self.coteacher.run(data = df, epoch = epoch)
            total_model1 += num_model1
            total_anomaly_model1 += anomaly_model_1

            total_model2 += num_model2
            total_anomaly_model2 += anomaly_model_2

            selection_df = pd.concat((selection_df, df))


            # 3. backward and optimization only in train
            if start_train:
                # model 1 update
                self.optim_schedule1.zero_grad()  # reset the gradient
                loss1 = mask_loss1.sum(dim=1) * model1_sel_tensor.to(self.device)
                loss1 = loss1.sum()
                loss1.backward()  # back propagation
                self.optim_schedule1.step_and_update_lr()
                # model 2 update
                self.optim_schedule2.zero_grad()  # reset the gradient
                loss2 = mask_loss2.sum(dim=1) * model2_sel_tensor.to(self.device)
                loss2 = loss2.sum()
                loss2.backward()  # back propagation
                self.optim_schedule2.step_and_update_lr()

        avg_loss = total_loss / totol_length  # after the epoch, calculate the avg loss of this epoch
        avg_vector_loss = total_vector_loss / totol_length

        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)

        print(f"Co-teaching model 1 selection size: {total_model1}, anomaly num: {total_anomaly_model1}, "
              f"model 2 selection size: {total_model2}, anomaly num: {total_anomaly_model2}, "
              f"remember rate: {self.coteacher.remember_rate}")
        print("Epoch: {} | phase: {}, learning rate model 1 ={}, learning rate model 2 = {}, avg seq loss={}".format(epoch, phase, lr1 , lr2 ,avg_loss))
        sample_metrics = {"train/epoch": epoch,
                          "train/model1_selection_num": total_model1,
                          "train/model2_selection_num": total_model2,
                          "train/model1_anomaly_num": total_anomaly_model1,
                          "train/model2_anomaly_num": total_anomaly_model2,
                          "train/model1_anomaly_ratio": round(float(total_anomaly_model1)/float(total_model1), 3),
                          "train/model2_anomaly_ratio": round(float(total_anomaly_model2)/float(total_model2), 3)
                          }

        selection_df.to_csv(self.model_dir+f"selection_{epoch}.csv")
        # wandb.log(sample_metrics, step=epoch)
        return avg_loss, total_dist


    def save_log(self, save_dir, surfix_log):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(save_dir + key + f"_{surfix_log}.csv",
                                            index=False)
            print("Log saved")
        except Exception as e:
            print(str(e))
            print("Failed to save logs")

    def save(self, save_dir="output/bert_trained.pth"):
        """
        Saving the current BERT model on file_path

        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        torch.save(self.model1, save_dir+"model1")
        torch.save(self.model2, save_dir+"model2")
        # self.bert.to(self.device)
        print(" Model Saved on:", save_dir)
        return save_dir

