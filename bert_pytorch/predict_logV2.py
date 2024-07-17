import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from bert_pytorch.dataset import WordVocab
from bert_pytorch.dataset import LogDataset
from bert_pytorch.dataset.sample import fixed_window
from bert_pytorch.dataset.utils import bert_read_file
from sklearn.metrics import roc_auc_score

#todo: give hints about abnormal events

def compute_anomaly(results, params, seq_threshold=0.5):
    is_logkey = params["is_logkey"]
    is_time = params["is_time"]
    total_anomalies = 0
    anomaly_seq_ids = []
    for seq_res in results:
        # label pairs as anomaly when over half of masked tokens are undetected
        if (is_logkey and seq_res["undetected_tokens"] > seq_res["masked_tokens"] * seq_threshold) or \
                (is_time and seq_res["num_error"]> seq_res["masked_tokens"] * seq_threshold) or \
                (params["hypersphere_loss_test"] and seq_res["deepSVDD_label"]):
            total_anomalies += 1
            anomaly_seq_ids.append(seq_res["seq_id"])
    return total_anomalies, anomaly_seq_ids



class PredictorV2():
    def __init__(self, options):
        self.data_dir = options["data_dir"]
        self.test_data_dir = options["test_data_dir"]
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.device = options["device"]
        self.seq_len = options["seq_len"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.num_candidates_ratio = options["num_candidates_ratio"]
        self.num_candidates= 0

        # stored data loader
        self.normal_dataloader = None
        self.abnormal_dataloader = None

        self.is_logkey = True
        self.is_time = False

        self.hypersphere_loss = False
        self.hypersphere_loss_test = False

        self.test_ratio = options["test_ratio"]
        self.mask_ratio = options["mask_ratio"]

        self.last_best_seq_th = 1.0
        
        self.log = {"epoch":[], "precision":[], "recall":[], "f-1":[], "TP": [], "TN": [], "FP": [], "FN": [], "accuracy":[], "auc": []}

    def detect_logkey_anomaly(self, masked_output, masked_label):
        num_undetected_tokens = 0
        output_maskes = []
        top_80_for_all = []
        for i, token in enumerate(masked_label):

            top_80 = torch.argsort(-masked_output[i])[:min(80, len(masked_output[i]))].tolist()
            top_80_for_all.append(top_80)
            #todo: automatically tune the number of candidates to get the best number and store it into wandb
            if token not in torch.argsort(-masked_output[i])[:self.num_candidates]:
                num_undetected_tokens += 1

        return num_undetected_tokens, [output_maskes, masked_label.cpu().numpy()], top_80_for_all

    def load_data(self, data_dir, file_name, vocab):

        seq_id_test, logkey_test, label_test, token_label_test= bert_read_file(data_dir + file_name + ".csv")

        # use 1/10 test data
        if self.test_ratio != 1:
            num_test = len(logkey_test)
            rand_index = torch.randperm(num_test)
            rand_index = rand_index[:int(num_test * self.test_ratio)] if isinstance(self.test_ratio, float) else rand_index[:self.test_ratio]
            logkey_test = logkey_test[rand_index]


        seq_dataset = LogDataset(logkey_test.tolist(),  vocab, seq_len=self.seq_len,
                                  predict_mode=True, mask_ratio=self.mask_ratio,
                                 label_corpus=label_test.tolist(), token_label_corpus=token_label_test.tolist(), seq_id= seq_id_test)

        # use large batch size in test data
        data_loader = DataLoader(seq_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                 collate_fn=seq_dataset.collate_fn)

        return data_loader

    def helper(self, model, data_loader, data_indicator):
        total_results = []
        output_results = []
        output_cls = []


        all_results = {"seq_id":[], "masked_token_num":[], "top_n_tokens":[], "token_label":[]}
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                data = {key: value.to(self.device) for key, value in data.items()}

                result = model(data["bert_input"])

                # mask_lm_output, mask_tm_output: batch_size x session_size x vocab_size
                # cls_output: batch_size x hidden_size
                # bert_label, time_label: batch_size x session_size
                # in session, some logkeys are masked

                mask_lm_output = result["logkey_output"]
                output_cls += result["cls_output"].tolist()

                # dist = torch.sum((result["cls_output"] - self.hyper_center) ** 2, dim=1)
                # when visualization no mask
                # continue

                all_results["seq_id"].extend(data["seq_id"].tolist())
                all_results["token_label"].extend(data["bert_label"].tolist())


                # loop though each sequence in batch
                for i in range(len(data["bert_label"])):
                    seq_results = {"num_error": 0,
                                   "undetected_tokens": 0,
                                   "masked_tokens": 0,
                                   "total_logkey": torch.sum(data["bert_input"][i] > 0).item(),
                                   "deepSVDD_label": 0,
                                   "seq_id": data["seq_id"][i].item()
                                   }

                    mask_index = data["bert_label"][i] > 0 # label of the masked token is itself, # the label of the unmasked token is 0
                    num_masked = torch.sum(mask_index).tolist()
                    seq_results["masked_tokens"] = num_masked
                    all_results["masked_token_num"].append(num_masked)

                    if self.is_logkey:
                        # num_undeteced: the number of masked tokens that are not predicted in top rank 15
                        # top_30_for_all_tokens: (token_num, 30)
                        num_undetected, output_seq, top_80_for_all_tokens = self.detect_logkey_anomaly(
                            mask_lm_output[i][mask_index], data["bert_label"][i][mask_index])
                        seq_results["undetected_tokens"] = num_undetected
                        all_results["top_n_tokens"].append(top_80_for_all_tokens)
                        output_results.append(output_seq)

                    total_results.append(seq_results)

        all_results = pd.DataFrame.from_dict(all_results)
        model_name = self.model_path.split("/")[-2]
        all_results.to_csv(self.test_data_dir + model_name + f"_{data_indicator}_test_results.csv")
        return total_results, output_cls

    def total_results_to_dataframe(self, total_results):
        result_dict = {"seq_id": [], "masked_tokens": [], "undetected_tokens":[]}
        for res in total_results:
            result_dict["seq_id"].append(res["seq_id"])
            result_dict["masked_tokens"].append(res["masked_tokens"])
            result_dict["undetected_tokens"].append(res["undetected_tokens"])

        result_df = pd.DataFrame.from_dict(result_dict)
        return result_df


    def predict(self, epoch = 0, model = None):
        if model is None:
            model = torch.load(self.model_path)

        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))
        num_candidates = int(self.num_candidates_ratio*len(self.vocab))

        best_num_candidates = 0.0
        best_F1_over_num_candidates = 0.0
        best_P_over_num_candidates = 0.0
        best_R_over_num_candidates = 0.0
        best_rocauc_over_num_candidates = 0.0
        best_TP = 0
        best_FP = 0
        best_TN = 0
        best_FN = 0
        best_thres = 0.0
        can_num_list = [num_candidates]
        # can_num_list = [max(num_candidates -5, 1), num_candidates,
        #                             min(num_candidates +5, len(self.vocab)), min(num_candidates+10, len(self.vocab))]
        # ratios = np.arange(0.01, 0.1, 0.01)
        # can_num_list = (ratios * len(self.vocab)).astype(int)

        for self.num_candidates in can_num_list:

            print(f"******** Candidate number as {self.num_candidates} *********")
            start_time = time.time()

            if self.normal_dataloader is None:
                self.normal_dataloader = self.load_data(self.test_data_dir, "test_normal", self.vocab)

            test_normal_results, test_normal_errors = self.helper(model, self.normal_dataloader, "normal")

            if self.abnormal_dataloader is None:
                self.abnormal_dataloader = self.load_data(self.test_data_dir, "test_abnormal", self.vocab)

            test_abnormal_results, test_abnormal_errors = self.helper(model, self.abnormal_dataloader, "abnormal")


            params = {"is_logkey": self.is_logkey, "is_time": self.is_time, "hypersphere_loss": self.hypersphere_loss,
                      "hypersphere_loss_test": self.hypersphere_loss_test}
            print(f"model path: {self.model_path}\n")
            threshold = 0.0
            FP, TP, TN, FN, P, R, F1 = self.fix_threshold(test_normal_results,
                                                                                test_abnormal_results,
                                                                                params=params,
                                                                                threshold= threshold)
            rocauc = self.get_rocauc_score(test_normal_results, test_abnormal_results, params=params)

            print(f"fix threshold: {threshold}")
            print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
            print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))

            best_num_candidates = self.num_candidates
            best_F1_over_num_candidates = F1
            best_P_over_num_candidates = P
            best_R_over_num_candidates = R
            best_rocauc_over_num_candidates = rocauc

            best_TP = TP
            best_FP = FP
            best_TN = TN
            best_FN = FN


            best_seq_th, FP, TP, TN, FN, P, R, F1 = self.find_best_threshold(test_normal_results,
                                                                                test_abnormal_results,
                                                                                params=params,
                                                                                seq_range=np.arange(0,1,0.1))
            self.last_best_seq_th = best_seq_th

            print("best threshold: {}".format(best_seq_th))
            print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
            print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))
            elapsed_time = time.time() - start_time
            print('elapsed_time: {}'.format(elapsed_time))


        self.log["epoch"].append(epoch)
        self.log["precision"].append(best_P_over_num_candidates)
        self.log["recall"].append(best_R_over_num_candidates)
        self.log["f-1"].append(best_F1_over_num_candidates)
        self.log["auc"].append(best_rocauc_over_num_candidates)
        self.log["TP"].append(best_TP)
        self.log["TN"].append(best_TN)
        self.log["FP"].append(best_FP)
        self.log["FN"].append(best_FN)

    def fix_threshold(self, test_normal_results, test_abnormal_results, params, threshold):

        FP, FP_seqs = compute_anomaly(test_normal_results, params, threshold)
        FP_df = pd.DataFrame()
        FP_df["false_positive_seq"] = FP_seqs
        model_name = self.model_path.split("/")[-2]
        TP, TP_seqs = compute_anomaly(test_abnormal_results, params, threshold )
        TP_df = pd.DataFrame()
        TP_df["false_positive_seq"] = TP_seqs

        TN = len(test_normal_results) - FP
        FN = len(test_abnormal_results) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        fix_thres_result = [ FP, TP, TN, FN, P, R, F1]
        return fix_thres_result


    def find_best_threshold(self, test_normal_results, test_abnormal_results, params,  seq_range):
        best_result = [0] * 9

        print(f"last best seq th :{self.last_best_seq_th}, exhaust search for best seq threshold.")
        for seq_th in seq_range:
            FP, FP_seqs = compute_anomaly(test_normal_results, params, seq_th)
            TP, TP_seqs  = compute_anomaly(test_abnormal_results, params, seq_th)

            if TP == 0:
                continue

            TN = len(test_normal_results) - FP
            FN = len(test_abnormal_results) - TP
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)

            print(f"Trial threshold: {seq_th}")
            print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
            print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))

            if F1 > best_result[-1]:
                best_result = [seq_th, FP, TP, TN, FN, P, R, F1]
                self.last_best_seq_th = seq_th

        return best_result


    def get_rocauc_score(self, test_normal_results, test_abnormal_results, params):

        undetected_ratios = []
        true_labels = np.concatenate((np.zeros(len(test_normal_results)), np.ones(len(test_abnormal_results))), axis= 0)
        for seq_res in test_normal_results:
            # label pairs as anomaly when over half of masked tokens are undetected
            undetected_ratio = seq_res["undetected_tokens"]/ seq_res["masked_tokens"] if seq_res["masked_tokens"] > 0 else 0.0
            undetected_ratios.append(undetected_ratio)

        for seq_res in test_abnormal_results:
            # label pairs as anomaly when over half of masked tokens are undetected
            undetected_ratio = seq_res["undetected_tokens"]/ seq_res["masked_tokens"] if seq_res["masked_tokens"] > 0 else 0.0
            undetected_ratios.append(undetected_ratio)

        auc_score = roc_auc_score(true_labels, undetected_ratios)

        print(f"Auc score: {auc_score}")
        return auc_score


