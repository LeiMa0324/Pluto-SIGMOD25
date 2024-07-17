import math
import time

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
import torch
from matplotlib import pyplot as plt
import copy
import torch.nn as nn
import seaborn as sns
import ast


from torch.utils.data import DataLoader
from bert_pytorch.dataset import LogDataset
import sys

sys.path.append("../..")
from sample.sampler.PLUTO.k_means_cluster import KMeans_Cluster
# from sample.sampler.MINE.pluto_resemble_and_individual import pluto_final
from sample.sampler.PLUTO.pluto import pluto
from sample.selection_measure import selection_measure
from sample.sampler.FINE.fine import fine
# sys.path.append("../..")

# todo: make if outlier cluster process a parameter, for depth < 3 and ratio >4, use outlier cluster process


class Sample_Selector():
    '''
    only store for training information
    '''
    def __init__(self,  anomaly_ratio: int,  train_dataset: LogDataset,
                 output_dir, selection_method='pluto',
                 source_depth = 0, select_depth = 1, p_thres = 0.5,
                 ITLM_percent = 0.6, outlier_cluster_thres = 80, if_process_outlier = True, cluster_num = 20 ,if_select=True):
        self.clustering_method = 'kmeans'
        self.selection_method = selection_method
        self.train_dataset = train_dataset
        self.source_data = None
        self.seq_info = None
        self.seq_to_cls_tensors = None
        # anomaly ratio of data set
        self.anomaly_ratio = anomaly_ratio
        self.if_process_outlier = if_process_outlier
        self.cluster_num = cluster_num
        # the epochs for the model to learn embeddings

        self.if_outlier_cluster_dict = None
        self.outlier_cluster_thres = outlier_cluster_thres
        self.if_select = if_select

        self.epoch = 0
        self.selection_epochs = 0
        self.selection_duration = 0
        self.recursive_selection_epochs = 0
        # cluster settings
        self.output_dir = output_dir
        self.source_depth = source_depth
        self.select_depth = select_depth

        # the history history of the selected samples
        self.historical_model_detected_anomaly = []
        self.selected_seq_id_to_label = {}
        self.selected_seq_id_to_count = {}
        self.history_sampled_anomaly_num = 0
        self.consensus_outlier_seq_ids = []
        self.hyper = False

        self.seq_to_first_vector_dict = {}
        self.seq_to_sec_vector_dict = {}

        self.hyper_loss = 0.0
        self.seq_to_center = {}

        self.last_selection = None

        self.p_thres = p_thres

        self.ITLM_percent = ITLM_percent


    def load_sequence_info(self, seq_info, epoch, code_str):

        self.epoch = epoch
        self.seq_info = seq_info
        self.seq_info.to_csv(self.output_dir+f"seq_info_epoch_{epoch}.csv")
        # the historical detected sequences
        self.historical_model_detected_anomaly.extend(self.seq_info[seq_info["detected"]==1]["seq_id"].to_numpy().tolist())
        self.historical_model_detected_anomaly = list(set(self.historical_model_detected_anomaly))

    def run(self, ori_train_data_loader: DataLoader, cur_train_data_loader: DataLoader,epoch: int, code_str: str, method: str):

        print(f"selecting samples using { self.selection_method }...")
        # no selection phase, combine selection with training, no outlier cluster process after the first selection

        start = time.time()

        if self.selection_method in ['pluto']:
            selection_seq_ids, selection =  self.pluto_select(epoch)
            self.calculate_weights(selection, default_weights=False)

        elif  self.selection_method  == 'fine+':
            selection_seq_ids, selection  = self.fine_select(epoch)
            # 5. calculate weights
            self.calculate_weights(selection, default_weights=True)
        elif  self.selection_method  in ['ITLM', 'ITLM_norm']:
            selection_seq_ids, selection  = self.ITLM_select(epoch)
            # 5. calculate weights
            self.calculate_weights(selection, default_weights=True)
        else:
            print(self.selection_method)
            raise NotImplementedError

        end = time.time()
        print(f"selection duration: {end-start} seconds")

        self.selection_epochs += 1
        selection.to_csv(self.output_dir+f"selection_{epoch}.csv")

        # 7. new data loader
        selection_indices = self.train_dataset.get_indices_of_seq_ids(selection_seq_ids)
        sampled_dataset = torch.utils.data.Subset(self.train_dataset, selection_indices)
        sampled_data_loader = torch.utils.data.DataLoader(sampled_dataset, batch_size=ori_train_data_loader.batch_size,
                                                          shuffle=True, num_workers=ori_train_data_loader.num_workers,
                                                          collate_fn = self.train_dataset.collate_fn, drop_last=True)
        return sampled_data_loader, selection_seq_ids

    def pluto_select(self, epoch: int):
        '''
        use the warm-up embeddings and the first time clustering for all pluto selections
        for each selection, select with model consensus and noisy_0 the original data set
        '''

        print("using pluto with fixed clustering.")
        # generate the source data to select from
        if self.source_data is None or len(self.source_data) == 0:
            self.source_data = self.seq_info
            self.source_data = self.source_data[
                (~self.source_data["seq_id"].isin(self.consensus_outlier_seq_ids))
                ].reset_index(drop=True)

            clusterer = KMeans_Cluster(cluster_num=self.cluster_num)
            self.source_data = clusterer.cluster(self.source_data)
            self.first_source_data = self.source_data

        source_data_ids = self.source_data["seq_id"]

        # source data are something from last round and except the consensus outliers
        # using the first time embedding and cluster to for pluto selection
        self.source_data = self.first_source_data[(~self.first_source_data["seq_id"].isin(self.consensus_outlier_seq_ids)) &
                                         (self.first_source_data["seq_id"].isin(source_data_ids))
                                         ].reset_index(drop=True)


        source_anomaly_ratio = float(len(self.source_data[self.source_data["seq_label"]==1]))/float(len(self.source_data))

        # only do one-time outlier cluster process when low anomaly ratio
        if self.if_process_outlier:
            if self.anomaly_ratio < 6:
                if_process_outlier = self.selection_epochs < 1
            else:
                if_process_outlier = True
        else:
            if_process_outlier = False
        #
        # # ablation study, not process outlier clusters
        # if_process_outlier = False

        selector = pluto(self.source_data, if_process_outlier = if_process_outlier, anomaly_ratio = 0.0,
                         outlier_cluster_thres=self.outlier_cluster_thres,
                         if_norm_loss = "norm" in self.selection_method, output_path=self.output_dir, if_select=self.if_select)

        # selector inliers and outliers
        selector_inliers, selection_time = selector.run(save_output=True, output_path=self.output_dir+f"source_{epoch}.csv")
        self.selection_duration += selection_time
        # store the outlier clusters
        self.if_outlier_cluster_dict = selector.if_outlier_cluster_dict
        print(f"outlier clusters: {self.if_outlier_cluster_dict}")

        self.seq_to_first_vector_dict = selector.seq_to_first_vector_dict
        self.seq_to_sec_vector_dict = selector.seq_to_sec_vector_dict

        selector_outliers = self.source_data[~self.source_data["seq_id"].isin(selector_inliers["seq_id"])]

        # model inliers and outliers: self.historical_model_detected_anomaly
        cur_consensus_outliers =  selector_outliers[selector_outliers["seq_id"].isin(self.historical_model_detected_anomaly)]
        cur_consensus_inliers = selector_inliers[~selector_inliers["seq_id"].isin(self.historical_model_detected_anomaly)]
        print(f"consensus inliers: {len(cur_consensus_inliers)}")

        # all history consensus outliers
        self.consensus_outlier_seq_ids.extend(cur_consensus_outliers["seq_id"].to_numpy().tolist())


        # 3. find the sampled dataset
        selection_seq_ids = cur_consensus_inliers["seq_id"]
        selection_seq_labels = cur_consensus_inliers["seq_label"]

        unique_normal_seq_num = len(cur_consensus_inliers[cur_consensus_inliers["seq_label"]==0]["seq"].unique())
        unique_abnormal_seq_num = len(cur_consensus_inliers[cur_consensus_inliers["seq_label"]==1]["seq"].unique())
        meas = selection_measure(self.source_data, cur_consensus_inliers) # measure the final selection
        meas.measure()


        # store historical sequences, historical anomaly nums
        self.count_sample_selection(selection_seq_ids, selection_seq_labels)


        # 6. log sampling metrics
        source_data_size = len(self.source_data)
        removed_by_model = selector_inliers[~selector_inliers["seq_id"].isin(cur_consensus_inliers["seq_id"])]
        removed_outlier_by_model = len(removed_by_model[removed_by_model["seq_label"]==1])
        removed_inlier_by_model = len(removed_by_model[removed_by_model["seq_label"] == 0])

        consensus_true_outlier = len(self.seq_info[(self.seq_info["seq_label"]==1) & (self.seq_info["seq_id"].isin(self.consensus_outlier_seq_ids))])

        sample_metrics = {"train/epoch": epoch,
                          "train/source_size":source_data_size,
                          "train/source_anomaly_ratio": source_anomaly_ratio,
                          "train/sample_num": meas.sel_sizes,
                          "train/sampled_anomaly_ratio": meas.sel_anomaly_ratios,
                          "train/sampled_anomaly_num": meas.sel_anomalies,
                          "train/sampled_unique_normal_num": unique_normal_seq_num,
                          "train/sampled_unique_abnormal_num": unique_abnormal_seq_num,
                          "train/consensus_anomaly": len(self.consensus_outlier_seq_ids),
                          "train/consensus_true_anomaly": consensus_true_outlier,
                          "train/avg_selection_duration": self.selection_duration/(self.selection_epochs+1)
                          }

        # wandb.log(sample_metrics, step=epoch)

        return selection_seq_ids, cur_consensus_inliers

    def get_source_data(self, if_simple=False):

        clusterer = KMeans_Cluster(cluster_num = self.cluster_num)


        if if_simple: # source data are from historical source data
            self.source_data = self.seq_info.reset_index(
                drop=True)
            self.source_data = clusterer.cluster(self.source_data)

        else: # source data are from all whole data

            if self.source_data is None or len(self.source_data)==0:
                self.source_data = self.seq_info

            source_data_ids = self.source_data["seq_id"]
            # source data is the sequence info which are
            # 1. not in the consensus outlier
            # 2. in the source data
            self.source_data = self.seq_info[(~self.seq_info["seq_id"].isin(self.consensus_outlier_seq_ids)) &
                                             (self.seq_info["seq_id"].isin(source_data_ids))
                                             ].reset_index(
                drop=True)

            self.source_data = clusterer.cluster(self.source_data)

    def fine_select(self,  epoch: int, if_log=True ):
        # 1. cluster the embeddings

        if self.source_data is None:
            # only cluster once
            clusterer = KMeans_Cluster(cluster_num=self.cluster_num)
            self.source_data  = clusterer.cluster(self.seq_info)
        else:
            # use the current features and the old cluster info
            temp = self.seq_info[self.seq_info["seq_id"].isin(self.source_data["seq_id"])].reset_index(drop=True)
            temp["cluster"] = temp["seq_id"].map(lambda s: self.source_data[self.source_data["seq_id"]==s]["cluster"].to_numpy()[0])
            self.source_data = temp

        if len(self.source_data) < 1000:
            selection = self.source_data
            meas_1 = selection_measure(selection, selection)
        else:
            fine_selector = fine(self.source_data )
            meas_1, selection = fine_selector.run(p_thres= self.p_thres)


        self.source_data = selection

        if if_log:
            sample_metrics = {"train/epoch": epoch,
                              "train/sampled_anomaly_ratio": meas_1.sel_anomaly_ratios,
                              "train/sampled_anomaly_num": meas_1.sel_anomalies,
                              "train/sample_num": meas_1.sel_sizes,
                              }
            # wandb.log(sample_metrics, step=epoch)

        return selection["seq_id"], selection


    def ITLM_select(self, epoch: int):

        k = int(self.ITLM_percent * len(self.seq_info))
        print(f"selecting {k} samples with ratio {self.ITLM_percent}...")
        if self.selection_method =='ITLM':
            self.seq_info.sort_values(by = ["seq_loss"])
            selection = self.seq_info[:k]
        elif  self.selection_method =='ITLM_norm':
            self.seq_info["seq_loss_norm"] = self.seq_info["seq_loss"]/self.seq_info["masked_num"]
            self.seq_info.sort_values(by = ["seq_loss_norm"])
            selection = self.seq_info[:k]
        else:
            raise NotImplementedError

        anomaly_num = len(selection[selection["seq_label"]==1])

        sample_metrics = {"train/epoch": epoch,
                          "train/sampled_anomaly_ratio": float(anomaly_num)/float(len(selection)),
                          "train/sampled_anomaly_num": anomaly_num,
                          "train/sample_num": len(selection)
                          }
        # wandb.log(sample_metrics, step=epoch)
        return selection["seq_id"], selection

    def count_sample_selection(self, sampled_seq_ids, sampled_seq_labels):
        for seq_id, seq_label in zip(sampled_seq_ids,sampled_seq_labels):
            if seq_id not in self.selected_seq_id_to_count.keys():
                self.selected_seq_id_to_label[seq_id] = seq_label
                self.selected_seq_id_to_count[seq_id] = 1
            else:
                self.selected_seq_id_to_count[seq_id] += 1

        anomaly_num = 0
        for key in self.selected_seq_id_to_label:
            if self.selected_seq_id_to_label[key] == 1:
                anomaly_num+=1

        self.history_sampled_anomaly_num = anomaly_num
        self.history_sampled_num = len(self.selected_seq_id_to_label)


    def select_on_history_votes(self, vote_thres):
        selected_seq_id = []
        for i, (seq_id, count) in enumerate(self.selected_seq_id_to_count.items()):
            if count > vote_thres:
                selected_seq_id.append(seq_id)
        return selected_seq_id

    def calculate_weights(self, selection, default_weights = False):

        self.seq_to_weights = {}
        sampled_seq_ids = selection["seq_id"].to_numpy()
        sampled_seq_labels = selection["seq_label"].to_numpy()
        if default_weights:
            for seq_id in sampled_seq_ids:
                self.seq_to_weights[seq_id] = 1.0
        else:
            # weight by the first score
            sum = 0.0
            id_normal = []
            id_abnormal = []
            weights_normal = []
            weights_abnormal = []
            for (seq_id, label) in zip(sampled_seq_ids, sampled_seq_labels):
                w = selection[selection["seq_id"]==seq_id]["first_score"].to_numpy()[0]
                self.seq_to_weights[seq_id]=w
                sum +=w
                if label==0:
                    id_normal.append(seq_id)
                else:
                    id_abnormal.append(seq_id)

            #normalize to mean 1
            for i, (seq_id, w) in enumerate(self.seq_to_weights.items()):
                # normalized and 1/N => 1/N*N ==> weighted average, not average
                # todo: very similar weights, change the weighting so it will make a larger difference
                self.seq_to_weights[seq_id] = (w*float(len(self.seq_to_weights))/sum)
                if seq_id in id_normal:
                    weights_normal.append(self.seq_to_weights[seq_id])
                else:
                    weights_abnormal.append(self.seq_to_weights[seq_id])

            df = selection[["seq_id", "seq_label",'first_score']]
            df.to_csv(self.output_dir+f"weights_epoch_{self.epoch}.csv")
            plt.figure()
            plt.hist(weights_normal, label="weights(normal)", color='steelblue', alpha = 0.5)
            plt.hist(weights_abnormal, label="weights(abnormal)", color='darkorange', alpha = 0.5)
            plt.yscale("log")
            plt.legend()
            plt.title(f"weights epoch {self.epoch}")
            plt.xlabel("weight")
            plt.savefig(self.output_dir+f"weights_epoch_{self.epoch}.png")


    def get_seq_weights(self, seq_ids):
        weights = []
        for id in seq_ids:
            weights.append(self.seq_to_weights[id])
        return np.array(weights)

    def get_seq_center_tensor(self, seq_ids):
        centers = None
        for id in seq_ids:
            if id not in self.seq_to_center.keys():
                continue
            if centers is None:
                centers = torch.from_numpy(self.seq_to_center[id]).unsqueeze(dim = 0)
            else:
                centers = torch.cat((centers, torch.from_numpy(self.seq_to_center[id]).unsqueeze(dim = 0)), dim=0)
        return centers

    def get_first_vector_tensor(self, seq_ids):
        vectors = None
        for id in seq_ids:
            if id not in self.seq_to_first_vector_dict.keys():
                continue
            if vectors is None:
                vectors = torch.from_numpy(self.seq_to_first_vector_dict[id]).unsqueeze(dim = 0)
            else:
                vectors = torch.cat((vectors, torch.from_numpy(self.seq_to_first_vector_dict[id]).unsqueeze(dim = 0)), dim=0)
        return vectors

    def parse_features(self, data):
        X = data["embedding"].to_numpy()
        if isinstance(X[0], str):
            X_features = []
            for x in X:
                emb_list = ast.literal_eval(x)
                X_features.append(np.array(emb_list))
            features = np.array(X_features)
        elif isinstance(X[0], list):
            Y = np.array([ np.array(x) for x in X])
            features = Y
        elif type(X[0]).__module__== np.__name__:
            features = X
        else:
            raise NotImplementedError

        return features




