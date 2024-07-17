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
from bert_pytorch.dataset import LogDataset
import torch.nn.functional as F
from bert_pytorch.dataset.utils import bert_read_file
from bert_pytorch.dataset import WordVocab

torch.cuda.empty_cache()

# todo: create a worm-up model to save time

class BERTTrainerV2:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction
    please check the details on README.md with simple example.
    """

    def __init__(self, bert: BERT, train_file_path: str, vocab_path: str, vocab_size: int,  model_dir,  num_candidates_ratio,
                 phase_checker: PhaseChecker,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 train_dataset: LogDataset =None, valid_dataset: LogDataset = None,
                 test_normal_dataloader: DataLoader =None, test_abnormal_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=5000,
                 with_cuda: bool = True,
                 robust_method = None,  store_embedding = False, sample_selector = None, seq_thres = 0.1):


        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.robust_method = robust_method

        self.hypersphere_loss = self.robust_method == "logbert"
        self.hyper_center = None
        self.nu = 0.25
        self.radius = 0

        self.train_file_path = train_file_path
        self.vocab_path = vocab_path
        self.vocab = WordVocab.load_vocab(self.vocab_path)

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
        self.bert = bert
        self.vocab_size = vocab_size

        self.phase_checker = phase_checker
        # Initialize the BERT Language Model, with BERT model

        if self.phase_checker.warmup_model_dir is not None and self.robust_method not in ['vanilla', 'logbert']:
            print(f"loading warm up model from {self.phase_checker.warmup_model_dir}...")
            self.model = torch.load(self.phase_checker.warmup_model_dir, map_location=self.device)
        else:
            print(f"initializing new bert model...")
            self.model = BERTLog(bert, vocab_size, output_attentions = False).to(self.device)

        if self.robust_method not in ['vanilla', 'logbert']:
            self.sample_selector = sample_selector

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optim = None
        self.optim_schedule = None
        self.init_optimizer()


        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # ignore the loss of label = 0
        self.criterion = nn.NLLLoss(ignore_index=0, reduce = False)  #reduce = False, return loss for each sample
        self.hyper_criterion = nn.MSELoss() # return sum loss of the whole mini batch

        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.epochs = self.phase_checker.epochs
        self.store_embedding = store_embedding

        self.seq_thres = seq_thres


    def init_optimizer(self):
        # Setting the Adam optimizer with h_vanilla-param
        self.optim = Adam(self.model.parameters(), lr= self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=self.warmup_steps, lr=self.lr)

    def train(self, epoch):
        if self.robust_method in ["vanilla", "logbert"]:

            return self.vanilla_iteration(epoch, self.train_data_loader, start_train=True)
        else:
            return self.interative_selection_iteration(epoch, self.train_data_loader, start_train=True)

    # vanilla iteration without robust method
    def vanilla_iteration(self, epoch, data_loader, start_train):
        str_code = "train" if start_train else "valid"
        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        total_loss = 0.0  # total loss
        total_logkey_loss = 0.0  # the loss of next log key prediction
        total_hyper_loss = 0.0
        total_dist = []

        for i, data in data_iter:

            result = self.model.forward(data["bert_input"].to(self.device))

            # return the prediction of the masked log key and the time interval
            mask_lm_output = result["logkey_output"]

            # 2-2. NLLLoss of predicting masked token word ignore_index = 0 to ignore unmasked tokens
            # since the last layer is a logsoftmax, here use NLlloss, if its a soft max layer, use CrossEntropy loss instead
            # mask_loss: a scalar loss of each sequence
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2),  data["bert_label"].to(self.device))

            # compute the weighted loss
            total_logkey_loss += mask_loss.sum()  # logkey loss is the sum of mask loss
            loss = mask_loss.sum()

            if self.hypersphere_loss:
                print("calculating hyper loss...")
                hyper_loss = self.hyper_criterion(result["cls_output"].squeeze(), self.hyper_center.expand(data["bert_input"].shape[0], -1))
                # version 2.0 https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
                dist = torch.sum((result["cls_output"] - self.hyper_center) ** 2, dim=1)
                total_dist += dist.cpu().tolist()

                total_hyper_loss += hyper_loss.item()

                # with deepsvdd loss
                loss = loss + 0.1 * hyper_loss

            total_loss += loss.item()

            # 3. backward and optimization only in train
            if start_train:
                self.optim_schedule.zero_grad()  # reset the gradient
                loss.backward()  # back propagation
                self.optim_schedule.step_and_update_lr()
                # self.optim.step()

        avg_loss = total_loss / totol_length  # after the epoch, calculate the avg loss of this epoch
        avg_hyper_loss = total_hyper_loss / totol_length  # after the epoch, calculate the avg loss of this epoch

        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        print("Epoch: {} | phase: {}, learning rate ={}, avg loss={}".format(epoch, str_code, lr , avg_loss))
        if self.hypersphere_loss:
            print(f"avg hyper loss : {avg_hyper_loss}")

        if self.store_embedding and start_train:
            print("saving embeddings...")
            sequence_info = self.get_embeddings(self.train_data_loader)
            embedding_path = self.model_dir+"embeddings/"
            if not os.path.exists(embedding_path):
                os.makedirs(embedding_path)
            sequence_info.to_csv(embedding_path+f"train_embedding_epoch{epoch}.csv", index=False)

        return avg_loss, total_dist


    def interative_selection_iteration(self, epoch, data_loader, start_train):

        str_code = "train" if start_train else "valid"
        start = time.strftime("%H:%M:%S")
        start_time = time.time()
        self.log[str_code]['time'].append(start)
        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        total_loss = 0.0  # total loss
        total_vector_loss = 0.0  # the loss of hypersphere
        total_dist = []

        lr = self.optim.state_dict()['param_groups'][0]['lr']
        self.log[str_code]['lr'].append(lr)

        phase = self.phase_checker.get_phase(epoch)
        selection_method = self.phase_checker.selection_method_dict[epoch]
        re_initialized = False

        # the first selection is before the training
        if self.phase_checker.is_selection_phase(epoch) and self.phase_checker.first_selection_phase(epoch):

            print(f"Start first selection at beginning of epoch {epoch}!")
            self.ori_train_data_loader = DataLoader(self.train_dataset,
                                                    batch_size=self.train_data_loader.batch_size,
                                                    num_workers=self.train_data_loader.num_workers,
                                                    collate_fn=self.train_dataset.collate_fn, drop_last=True,
                                                    shuffle=True)
            # get sequence embeddings and if detected anomalies,  load them into sample selector
            sequence_info = self.get_embeddings(self.ori_train_data_loader)
            self.sample_selector.load_sequence_info(sequence_info, epoch, str_code)
            self.train_data_loader, selection_seq_ids = self.sample_selector.run(self.ori_train_data_loader, self.train_data_loader, epoch, str_code, selection_method)


            print(f"****************** Re-initilize the Model After first selection! ********************")
            self.bert = BERT(self.vocab_size, hidden=self.bert.hidden, n_layers=self.bert.n_layers,
                             attn_heads=self.bert.attn_heads)
            self.model = BERTLog(self.bert, self.vocab_size, output_attentions=False).to(self.device)
            self.init_optimizer()
            data_iter = enumerate(self.train_data_loader)
            totol_length = len(data_loader)
            re_initialized = True

        for i, data in data_iter:
            result = self.model.forward(data["bert_input"].to(self.device))

            mask_lm_output = result["logkey_output"]

            # shape (32, 512)
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"].to(self.device))

            loss = mask_loss.sum()
            total_loss += loss.item()

            # 3. backward and optimization only in train
            if start_train:
                self.optim_schedule.zero_grad()  # reset the gradient
                loss.backward()  # back propagation
                self.optim_schedule.step_and_update_lr()


        avg_loss = total_loss / totol_length  # after the epoch, calculate the avg loss of this epoch
        avg_vector_loss = total_vector_loss / totol_length

        end_time = time.time()
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)


        print("Epoch: {} | phase: {}, training data size ={}, learning rate ={}, avg seq loss={}, avg vector loss = {}"
              .format(epoch, phase, len(self.train_data_loader.dataset), lr , avg_loss, avg_vector_loss))
        print(f"training elapse time :{end_time - start_time} seconds")

        if self.phase_checker.is_selection_phase(epoch):
            if phase =='selection' and (not self.phase_checker.first_selection_phase(epoch)):
                print(f"Start {phase} at end of epoch {epoch}!")
                # get sequence embeddings and if detected anomalies,  load them into sample selector
                sequence_info = self.get_embeddings(self.ori_train_data_loader)
                self.sample_selector.load_sequence_info(sequence_info, epoch, str_code)

                self.train_data_loader, self.selection_seq_ids = self.sample_selector.run(self.ori_train_data_loader, self.train_data_loader, epoch, str_code, selection_method)

            else:
                print(f"Freeze selection at end of epoch {epoch}!")

            if (self.phase_checker.selection_switching_to_retrain(epoch) )\
                    and (not re_initialized):

                # if self.robust_method=='pluto_ITLM_fix_cluster' and self.phase_checker.is_last_selection(epoch):
                #     print(f"******************  Re-generate vocab Before Retraining!  ********************")
                #     self.re_generate_vocab_and_train_set(self.selection_seq_ids)

                print(f"******************  Re-initilize the Model Before Retraining!  ********************")
                self.bert = BERT(self.vocab_size, hidden=self.bert.hidden, n_layers=self.bert.n_layers,
                                 attn_heads=self.bert.attn_heads)
                self.model = BERTLog(self.bert, self.vocab_size, output_attentions=False).to(self.device)
                self.init_optimizer()

        return avg_loss, total_dist

    def get_embeddings(self,  data_loader):

        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        sequence_info = pd.DataFrame(columns=["seq_label", "seq_id", "embedding",  "detected", "seq"])
        embedding_info = []
        seq_label = []
        seq_id = []
        seqs = []
        seq_loss = []
        total_detected = []
        masked_nums = []
        with torch.no_grad():

            for i, data in data_iter:

                result = self.model.forward(data["bert_input"].to(self.device))

                mask_lm_output = result["logkey_output"]

                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"].to(self.device))
                # detect if outlier
                batch_detected = self.compute_anomaly(data, mask_lm_output)

                # store the info of the sequence for sample selection
                seqs_lists = data["bert_ori_input"].detach().cpu().numpy().tolist()
                seqs.extend([ ','.join(str(e) for e in list) for list in seqs_lists])
                embeddings = result["cls_output"].detach().cpu().numpy().tolist()
                embedding_info.extend(embeddings)
                seq_label.extend(data["label"].detach().cpu().numpy().tolist())
                seq_id.extend(data["seq_id"].detach().cpu().numpy().tolist())
                seq_loss.extend(mask_loss.sum(dim=1).detach().cpu().numpy().tolist())
                total_detected.extend(batch_detected)
                mask_num = (torch.sum(data["bert_label"]>0, dim = 1)).detach().cpu().numpy().tolist()
                masked_nums.extend(mask_num)


        sequence_info["embedding"] = embedding_info
        sequence_info["seq"] = seqs
        sequence_info["seq_id"] = seq_id
        sequence_info["seq_label"] = seq_label
        sequence_info["seq_loss"] = seq_loss
        sequence_info["detected"] = sequence_info["seq_id"].isin(total_detected).astype(int)
        sequence_info["masked_num"] = masked_nums
        return sequence_info

    def detect_logkey_anomaly(self, masked_output, masked_label):
        num_undetected_tokens = 0
        output_maskes = []
        for i, token in enumerate(masked_label):

            if token not in torch.argsort(-masked_output[i])[:self.num_candidates]:
                num_undetected_tokens += 1

        return num_undetected_tokens, [output_maskes, masked_label.cpu().numpy()]

    def compute_anomaly(self,batch, mask_lm_output):
        detected_anomalies = []


        for i in range(len(batch["bert_label"])):
            seq_id = batch["seq_id"][i].item()
            mask_index = batch["bert_label"][i] > 0
            num_masked = torch.sum(mask_index).tolist()
            masked_tokens = num_masked

            # num_undeteced: the number of masked tokens that are not predicted in top rank 15
            num_undetected, _ = self.detect_logkey_anomaly(
                mask_lm_output[i][mask_index], batch["bert_label"][i][mask_index])

            # label pairs as anomaly when over half of masked tokens are undetected
            if (num_undetected> masked_tokens * self.seq_thres):
                detected_anomalies.append(seq_id)

        return detected_anomalies

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
        torch.save(self.model, save_dir)
        # self.bert.to(self.device)
        print(" Model Saved on:", save_dir)
        return save_dir

    @staticmethod
    def get_radius(dist: list, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist), 1 - nu)