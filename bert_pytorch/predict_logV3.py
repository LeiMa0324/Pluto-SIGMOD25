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
from sklearn.metrics import roc_auc_score, average_precision_score


class PredictorV3():
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
        
        self.log = {"epoch":[], "precision":[], "recall":[], "f-1":[], "TP": [], "TN": [], "FP": [], "FN": [], "accuracy":[], "auc": []}



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


    def positive_test_with_top_t_v2(self, results, top_t_list):

        pred_result = {"seq_id": [], "top_t": [], "positive": [], "undetected_ratio": []}
        print(f"Positive test with top k list {top_t_list}...")

        results["token_label"] = results["token_label"].map(lambda tl: np.array(tl))
        results["top_n_tokens"] = results["top_n_tokens"].map(lambda tl: np.array(tl))
        results["mask_token_index"] = results["token_label"].map(lambda tl: np.array(tl) > 0)
        results["mask_token_label"] = results.apply(lambda row: row["token_label"][row["mask_token_index"]])
        results["mask_top_n_tokens"] = results.apply(lambda row: row["top_n_tokens"][row["mask_token_index"]])

        for top_t in top_t_list:

            results["mask_top_t_tokens"] = results["mask_top_n_tokens"].map(lambda top_n: top_n[:top_t])
            results["positive_token_num"] = results.apply(lambda row: np.sum(np.array([1 if l in row["mask_top_t_tokens"][i] else 0
                                                                                for i, l in enumerate(row["mask_token_label"])])))
            results["positive_flag"] = results["positive_token_num"] > 0
            results["undetected_ratio"] = results["positive_flag"] / results["masked_token_num"]
            results["undetected_ratio"]  = results["undetected_ratio"] .replace(np.inf, 0.0)

            pred_result["seq_id"].extend(results["seq_id"].tolist())
            pred_result["top_t"].extend((np.zeros(len(results)) + top_t).tolist())
            pred_result["positive"].extend(results["positive_flag"].tolist())
            pred_result["extend"].extend(results["undetected_ratio"].tolist())


        pred_result = pd.DataFrame.from_dict(pred_result)
        print(f"Positive test done!")
        return pred_result

    def positive_test_with_top_t(self, results, top_t_list):

        pred_result = {"seq_id": [], "top_t": [], "positive": [], "undetected_ratio": []}
        print(f"Positive test with top k list {top_t_list}...")

        with tqdm(total=results.shape[0]) as pbar:
            for index, row in results.iterrows():
                pbar.update(1)
                masked_index = np.array(row["token_label"]) > 0
                masked_top_n_tokens = np.array(row["top_n_tokens"])[masked_index]
                masked_token_labels = np.array(row["token_label"])[masked_index]
                masked_token_num = row["masked_token_num"]

                assert len(masked_top_n_tokens) == len(masked_token_labels) == masked_token_num

                for top_t in top_t_list:
                    anomaly_flag = 0
                    wrong_token_num = 0
                    top_t_tokens = np.array([token[:top_t] for token in masked_top_n_tokens])

                    for (t_candidates, t) in zip(top_t_tokens, masked_token_labels):
                        if t not in t_candidates:
                            wrong_token_num += 1

                    if wrong_token_num > 0:
                        anomaly_flag = 1

                    undetected_ratio = wrong_token_num / masked_token_num if masked_token_num > 0 else 0.0

                    pred_result["seq_id"].append(index)
                    pred_result["top_t"].append(top_t)
                    pred_result["positive"].append(anomaly_flag)
                    pred_result["undetected_ratio"].append(undetected_ratio)

        pred_result = pd.DataFrame.from_dict(pred_result)
        print(f"Positive test done!")
        return pred_result

    def calculating_metrics(self,top_t_range, test_abnormal_pred_result, test_normal_pred_result ):

        print(f"Calculating metrics with top k list {top_t_range}...")
        metric_result_df = pd.DataFrame()
        for top_t in top_t_range:
            print(f"======================= top {top_t} tokens =======================")
            seq_thres_range = np.arange(0.0, 0.1, 0.1)
            anomaly_num = len(test_abnormal_pred_result[test_abnormal_pred_result["top_t"] == top_t])
            nomaly_num = len(test_normal_pred_result[test_normal_pred_result["top_t"] == top_t])

            TP = test_abnormal_pred_result[test_abnormal_pred_result["top_t"] == top_t]["positive"].sum()
            FN = len(test_abnormal_pred_result[test_abnormal_pred_result["top_t"] == top_t]) - TP

            FP = test_normal_pred_result[test_normal_pred_result["top_t"] == top_t]["positive"].sum()
            TN = len(test_normal_pred_result[test_normal_pred_result["top_t"] == top_t]) - FP

            P = np.around(TP / (TP + FP) * 100, 2)
            R = np.around(TP / (TP + FN) * 100, 2)
            F1 = np.around(2 * P * R / (P + R), 2)

            true_labels = np.concatenate(
                (np.zeros(nomaly_num), np.ones(anomaly_num)), axis=0)
            undetected_ratios = np.concatenate(
                (
                 test_normal_pred_result[test_normal_pred_result["top_t"] == top_t]["undetected_ratio"].to_numpy(),
                test_abnormal_pred_result[test_abnormal_pred_result["top_t"] == top_t]["undetected_ratio"].to_numpy(),),
                axis=0)
            auc_score = round(roc_auc_score(true_labels, undetected_ratios), 4)
            aupr = round(average_precision_score(true_labels, undetected_ratios), 4)

            print(f"Top K: {top_t}, TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}, "
                  f"Precision: {P}%, Recall: {R}%, F-1: {F1}%, Auc score: {auc_score}")

            df = pd.DataFrame()
            df["top_t"] = np.zeros(len(seq_thres_range)) + top_t

            df["TP"] = TP
            df["FN"] = FN
            df["TN"] = TN
            df["FP"] = FP
            df["Precision"] = P
            df["Recall"] = R
            df["F1"] = F1
            df["auc"] = np.zeros(len(seq_thres_range)) + auc_score
            df["aupr"] = np.zeros(len(seq_thres_range)) + aupr

            metric_result_df = pd.concat((metric_result_df, df))
        print(f"Calculating metrics done!")
        return metric_result_df

    def helper(self, model, data_loader, top_t_range):
        output_cls = []
        all_results = {"seq_id":[], "masked_token_num":[], "top_n_tokens":[], "token_label":[]}
        print("Obtaining raw predictions...")
        with torch.no_grad():

            for idx, data in enumerate(data_loader):
                data = {key: value.to(self.device) for key, value in data.items()}

                result = model(data["bert_input"])

                logkey_output = result["logkey_output"]
                mask_index = data["bert_label"] > 0

                output_cls += result["cls_output"].tolist()

                all_results["seq_id"].extend(data["seq_id"].tolist())
                all_results["token_label"].extend(data["bert_label"].tolist())
                all_results["masked_token_num"].extend(torch.sum(mask_index, -1).tolist())
                all_results["top_n_tokens"].extend(torch.argsort(-logkey_output, dim = -1)[:max(top_t_range)].tolist())


        all_results = pd.DataFrame.from_dict(all_results)
        # pred_result = {"seq_id": [], "top_t": [], "positive": [], "undetected_ratio": []}
        pred_result = self.positive_test_with_top_t(all_results, top_t_range)
        print("Raw predictions obtained!")
        return pred_result


    def predict(self, epoch = 0, model = None):
        if model is None:
            model = torch.load(self.model_path)

        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))
        model_name = self.model_path.split("/")[-2]
        self.num_candidates = int(self.num_candidates_ratio*len(self.vocab))

        start_time = time.time()

        top_t_range = np.arange(1, min(81, len(self.vocab))) # the range of top t

        if self.normal_dataloader is None:
            self.normal_dataloader = self.load_data(self.test_data_dir, "test_normal", self.vocab)

        test_normal_pred_result = self.helper(model, self.normal_dataloader,   top_t_range)
        normal_scores_dir = self.test_data_dir + model_name + f"_normal_scores.csv"
        test_normal_pred_result.to_csv(normal_scores_dir)

        if self.abnormal_dataloader is None:
            self.abnormal_dataloader = self.load_data(self.test_data_dir, "test_abnormal", self.vocab)

        test_abnormal_pred_result = self.helper(model, self.abnormal_dataloader, top_t_range)
        abnormal_scores_dir = self.test_data_dir + model_name + f"_abnormal_scores.csv"
        test_abnormal_pred_result.to_csv(abnormal_scores_dir)

        metric_result_df = self.calculating_metrics(top_t_range, test_abnormal_pred_result, test_normal_pred_result)


        metric_dir = self.test_data_dir + model_name + f"_metric_results.csv"
        metric_result_df.to_csv(metric_dir)
        print(f"Calculated metric saved to {metric_dir}.")

        print("=======================================================")
        print(f"******** Candidate number as {self.num_candidates} *********")

        TP = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["TP"].iloc[0]
        TN = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["TN"].iloc[0]
        FP = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["FP"].iloc[0]
        FN = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["FN"].iloc[0]
        P = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["Precision"].iloc[0]
        R = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["Recall"].iloc[0]
        F1 = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["F1"].iloc[0]
        rocauc = metric_result_df[metric_result_df["top_t"] == self.num_candidates]["auc"].iloc[0]
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))

        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))


        self.log["epoch"].append(epoch)
        self.log["precision"].append(P)
        self.log["recall"].append(R)
        self.log["f-1"].append(F1)
        self.log["auc"].append(rocauc)
        self.log["TP"].append(TP)
        self.log["TN"].append(TN)
        self.log["FP"].append(FP)
        self.log["FN"].append(FN)




