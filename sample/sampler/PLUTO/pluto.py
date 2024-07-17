import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import ast
import seaborn as sns
import warnings
from sample.sampler.PLUTO.k_means_cluster import KMeans_Cluster
from sample.selection_measure import selection_measure
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances
from sample.sampler.PLUTO.facility_location import FacilityLocation
from sample.sampler.PLUTO.lazyGreedy import *

warnings.filterwarnings('ignore')

'''
allow missing cluster, every thing associate with a cluster is stored in a dict
r/(1-r) = alpha/Dom
'''

class pluto():
    def __init__(self, data, outlier_cluster_thres = 80, if_process_outlier = True, re_clustering = False,
                 anomaly_ratio = 0.0, p_thres = None, remove_large_loss = True, store_clustered= False,
                 output_path = None,  random_state = 1234, if_norm_loss = False, if_select=True):
        self.data = data
        self.anomaly_ratio = anomaly_ratio
        self.remove_large_loss = remove_large_loss
        self.if_norm_loss = if_norm_loss
        self.features = None

        self.singular_value_dict = None
        self.first_singular_vector_dict = None
        self.sec_singular_vector_dict = None

        self.seq_to_first_vector_dict = None

        self.cluster_dominance_dict = None

        self.clean_labels = None
        self.est_anomaly_ratio_dict = None
        self.if_outlier_cluster_dict = None
        self.outlier_cluster_thres = outlier_cluster_thres

        self.if_process_outlier = if_process_outlier
        self.if_select = if_select
        self.p_thres = p_thres
        self.store_clustered = store_clustered
        self.output_path = output_path
        self.re_clustering = re_clustering
        self.random_state = random_state
        self.cluster()
        self.calculate_anomalies()
        self.parse_features()
        print("using pluto_crust")

        self.dom_ori = []
        self.dom_before_exclusion = []
        self.dom_after_exclusion = []

    def cluster(self):
        if "cluster" not in self.data.columns or self.re_clustering:
            cluster_num = 20
            start = time.time()
            print(f"Begin clustering for {cluster_num} using seed {self.random_state}...")
            clusterer = KMeans_Cluster(cluster_num)
            end = time.time()
            print(f"pluto_final.cluster runtime: {end - start} seconds")
            self.data = clusterer.cluster(self.data, random_state= self.random_state)
            if self.store_clustered:
                self.data.to_csv(self.output_path)

    def reset(self, data):
        self.data = data
        self.calculate_anomalies()
        self.features = None

        self.singular_value_dict = None
        self.first_singular_vector_dict = None
        self.sec_singular_vector_dict = None

        self.seq_to_first_vector_dict = None
        self.seq_to_sec_vector_dict = None

        self.cluster_dominance_dict = None

        self.clean_labels = None
        self.est_anomaly_ratio_dict = None
        self.if_outlier_cluster_dict = None

        self.parse_features()

    def calculate_anomalies(self):
        if self.anomaly_ratio ==0.0:
            self.anomaly_num = len(self.data[self.data["seq_label"]==1])
            print(f"using real anomaly ratio: {self.anomaly_num /len(self.data)}")
        else:
            self.anomaly_num = int(len(self.data)*self.anomaly_ratio/100)
            print(f"using defined anomaly ratio: {self.anomaly_ratio}")

    def parse_features(self):
        X = self.data["embedding"].to_numpy()
        if isinstance(X[0], str):
            X_features = []
            for x in X:
                emb_list = ast.literal_eval(x)
                X_features.append(np.array(emb_list))
            self.features = np.array(X_features)
        elif isinstance(X[0], list):
            Y = np.array([ np.array(x) for x in X])
            self.features = Y
        elif type(X[0]).__module__== np.__name__:
            self.features = X
        else:
            raise NotImplementedError

    def get_singular_vector(self, vector_ranks ):
        '''
        To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
        features: hidden feature vectors of data (numpy)
        labels: correspoding label list
        '''
        singular_value_dict = {}

        singular_vector_dict = {}
        seq_to_vector_dict = {}
        rank_to_result = {}

        for c in self.data["cluster"].unique():
            cluster = self.data[self.data["cluster"]==c]
            _, sigma, v = np.linalg.svd(self.features[self.data["cluster"]==c])
            for rank in vector_ranks:
                singular_vector_dict[c] = v[rank] # the first singular vector
                seq_to_vector_dict.update({seq: v[rank] for seq in cluster["seq_id"]})
                rank_to_result[rank] = {"singular_vector_dict": singular_vector_dict,
                                        "seq_to_vector_dict": seq_to_vector_dict}

            singular_value_dict[c] = sigma


        return  singular_value_dict, rank_to_result

    def run(self, depth = 1, save_output = False, output_path = None):

        start = time.time()
        ##################### step 1: SVD computation #####################
        start = time.time()
        self.singular_value_dict, rank_to_result = self.get_singular_vector(vector_ranks= [0,1])
        end1 = time.time()
        print(f"pluto_final.SVD runtime: {end1 - start} seconds")

        self.first_singular_vector_dict = rank_to_result[0]["singular_vector_dict"]
        self.seq_to_first_vector_dict  = rank_to_result[0]["seq_to_vector_dict"]
        self.sec_singular_vector_dict = rank_to_result[1]["singular_vector_dict"]
        self.seq_to_sec_vector_dict  = rank_to_result[1]["seq_to_vector_dict"]


        ##################### step 2: pollution estimation #####################

        end2 = time.time()
        self.est_anomaly_ratio_dict, self.cluster_mean_loss_dict = self.estimate_anomaly_ratio_and_mean_loss()


        if self.if_process_outlier and self.if_outlier_cluster_dict is None:
            print("detecting outlier clusters...")
            self.if_outlier_cluster_dict = self.get_outlier_clusters()
        end3 = time.time()
        print(f"pluto_final.pollution_estimator runtime: {end3 - end2} seconds")

        ##################### step 3: subset selection #####################
        end4 = time.time()
        self.data["first_score"] = self.get_score(self.first_singular_vector_dict)
        self.data["sec_score"] = self.get_score(self.sec_singular_vector_dict)

        # directly discarding high pollution clusters
        normal_selection_ids = self.fit_mixture_crust(score_rank=0, ratios_dict= self.est_anomaly_ratio_dict)
        # abnormal_selection_ids, _ = self.fit_mixture_old(score_rank=1,
        #                                                  ratios_dict={c: 1-r for c, r in self.est_anomaly_ratio_dict.items() })



        selection = self.data[self.data["seq_id"].isin(normal_selection_ids)]
        meas_1 = selection_measure(self.data, selection, self.if_outlier_cluster_dict, self.cluster_dominance_dict)
        meas_1.measure()
        meas_1.print_measures()

        #
        if self.remove_large_loss:
            print("removing large loss samples...")
            large_loss_selection_ids = self.large_loss_selection(ratios_dict = {c: 1-r for c, r in self.est_anomaly_ratio_dict.items() })
            selection = selection[~selection["seq_id"].isin(large_loss_selection_ids)]
            meas_4 = selection_measure(self.data, selection, self.if_outlier_cluster_dict, ori_domiance=self.cluster_dominance_dict)
            meas_4.measure()
            meas_4.print_measures()
            self.selection = selection

        self.data["selected"] = self.data["seq_id"].isin(selection["seq_id"])
        end5 = time.time()
        print(f"pluto_final.subset_selection runtime: {end5 - end4} seconds")

        selection_time = end5 - start
        print(f"pluto_final.total runtime: {end5 - start} seconds")

        if self.if_outlier_cluster_dict is not None:
            self.data["outlier_cluster"] = self.data["cluster"].map(lambda c: self.if_outlier_cluster_dict[c])
        self.data["cluster_anomaly_ratio"] = self.data["cluster"].map(lambda c: meas_1.ori_cluster_anomaly_ratios[c])
        if save_output:
            self.data.to_csv(output_path)

        end = time.time()
        print(f"pluto_final.run time duration: {end-start}")
        return selection, selection_time



    def calculate_cluster_p_threshold(self, ratio, probs):

        ratio = 1 if ratio > 1 else ratio
        ratio = 0 if ratio < 0 else ratio
        if int(ratio*100)>100 or int(ratio*100)<0:
            print(f"invalid ratio: {(ratio*100)}")
        return np.percentile(probs, int(ratio*100))


    def estimate_anomaly_ratio_and_mean_loss(self):

        cluster_dominance = {}
        max_dom = 0
        cluster_size_dict = {}
        cluster_loss_dict = {}

        for c in self.data["cluster"].unique():
            if len(self.singular_value_dict[c])>1:
                dom = float(self.singular_value_dict[c][0]) / float(self.singular_value_dict[c][1])
            else:
                dom = 0.0

            max_dom = max(dom, max_dom)
            mean_loss = self.data[self.data["cluster"]==c]["seq_loss"].mean()
            cluster_loss_dict[c] = mean_loss
            cluster_dominance[c] = dom
            cluster_size_dict[c] =len(self.data[self.data["cluster"]==c])


        self.cluster_dominance_dict = cluster_dominance
        print(self.cluster_dominance_dict)
        self.data["dominance"] = self.data["cluster"].map(lambda x: cluster_dominance[x])

        # dom_score_dict = {c: 1 - dom / max_dom for c, dom in self.cluster_dominance_dict.items()}
        dom_score_dict = {c: 1/dom if dom > 0 else 0.0 for c, dom in self.cluster_dominance_dict.items()}
        scale = self.anomaly_num/np.sum(np.array(list(dom_score_dict.values()))* np.array(list(cluster_size_dict.values())))
        est_anomaly_ratio_dict = {c: scale/(1/dom_score+scale) if dom_score>0.0 else 0.0 for c, dom_score in dom_score_dict.items()}

        return est_anomaly_ratio_dict, cluster_loss_dict

    def get_score(self, singular_vector_dict,  normalization=True):
        '''
        Calculate the score providing the degree of showing whether the data is noisy_0 or not.
        by the inner product of the sigular vector and the feature vector
        '''
        if normalization:
            scores = [np.abs(np.inner(singular_vector_dict[self.data["cluster"][indx]], feat / np.linalg.norm(feat))) for indx, feat
                      in enumerate(self.features)]
        else:
            scores = [np.abs(np.inner(singular_vector_dict[self.data["cluster"][indx]], feat)) for indx, feat in
                      enumerate(self.features)]

        return np.array(scores)

    def save_sampled_train(self, train_path, output_path):

        train = pd.read_csv(train_path)
        sampled_train = train[train["seq_id"].isin(self.selection["seq_id"])]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        sampled_train.to_csv(output_path+"traintrain.csv")

    def exclude_medoids(self, cluster_df, cluster_id, selected_seq_ids):
        anomaly_num = len(cluster_df[ (cluster_df['seq_id'].isin(selected_seq_ids)) & (cluster_df["seq_label"]==1)])
        print(f"before exlcusion: medoids num: {len(selected_seq_ids)}, anomaly num: {anomaly_num}")
        scores = cluster_df["sec_score"]
        scores = np.ravel(scores).astype(float).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)

        gmm.fit(scores)
        prob = gmm.predict_proba(scores)
        index = cluster_df.index
        prob = prob[:, gmm.means_.argmax()]  # prob of larger mean distribution

        ratios_dict = {c: 1 - r for c, r in self.est_anomaly_ratio_dict.items()}
        # percent ratio of values are smaller than p_thres
        p_thres = self.calculate_cluster_p_threshold(ratios_dict[cluster_id], prob)
        if_select = prob > p_thres

        bad_medoids_ids = cluster_df[if_select]["seq_id"].to_numpy()
        selected_seq_ids = np.setdiff1d(selected_seq_ids, bad_medoids_ids)
        anomaly_num = len(cluster_df[ (cluster_df['seq_id'].isin(selected_seq_ids)) & (cluster_df["seq_label"]==1)])
        print(f"after exlcusion: medoids num: {len(selected_seq_ids)}, anomaly num: {anomaly_num}")
        return selected_seq_ids

    def fit_mixture_crust(self, score_rank, ratios_dict):
        '''
        Assume the distribution of scores: bimodal gaussian mixture model

        return noisy_0 labels
        that belongs to the noisy_0 cluster by fitting the score distribution to GMM
        '''

        selection_ids = []
        indices = []
        large_mean_probs = []

        ssets = []
        weights = []

        for cls in self.data["cluster"].unique():

            if self.if_process_outlier and self.if_outlier_cluster_dict[cls]:
                continue

            cluster = self.data[self.data["cluster"] == cls]
            sample_ids = cluster["seq_id"].to_numpy()

            if self.if_select:
                print("with selection for slightly polluted clusters")
                #
                # mark: CRUST use estimated gradients, we use embeddings instead
                cluster_features = self.features[cluster.index]

                # calculate the pair wise distances in a matrix
                dists = pairwise_distances(cluster_features)
                # the size of each cluster as weight

                # Distance threshold (i.e. radius) in calculating clusters.
                dist_thres = 2.0
                weight = np.sum(dists < dist_thres, axis=1)
                # V: number of data points
                V = range(len(cluster_features))
                # F: facility location problem
                F = FacilityLocation(V, D=dists)
                # Ratio for number of facilities.

                selection_size = int((1-ratios_dict[cls]) * len(V))
                # return the results of selected data points
                sset, vals = lazy_greedy_heap(F, V, selection_size)
                # add the weights of the selected data points
                weights.extend(weight[sset].tolist())

                if len(sset) > 0:
                    sset = sample_ids[np.array(sset)]
                else:
                    continue
                #
                # _, sigma, v = np.linalg.svd(self.features[self.data["seq_id"].isin(sset)])
                # dom_before_exclude = sigma[0]/sigma[1]
                # self.dom_before_exclusion.append(dom_before_exclude)

                sset = self.exclude_medoids(cluster, cls, sset)
                #
                # # new dom of selection
                # _, sigma, v = np.linalg.svd(self.features[self.data["seq_id"].isin(sset)])
                # dom_after_exclude = sigma[0]/sigma[1]
                #
                # self.dom_after_exclusion.append(dom_after_exclude)
                # self.dom_ori.append(cluster.iloc[0]['dominance'] )
                # print(f"original dom: {cluster.iloc[0]['dominance'] }, selection dom before exclusion: {dom_before_exclude}, selection dom after exclusion: {dom_after_exclude},")

                ssets += list(sset)
            else:
                print("without selection for slightly polluted clusters")
                ssets += sample_ids.tolist()

        # weight_dict = {s: weights[i] for (i, s) in enumerate(ssets)}

        selection = self.data[self.data["seq_id"].isin(ssets)]
        # selection["weight"] = selection["seq_id"].map(lambda s: weight_dict[s])


        return selection["seq_id"].to_numpy()


    def large_loss_selection(self, ratios_dict):
        selection_ids = []
        #todo: using normalized loss
        for cls in self.data["cluster"].unique():
            cluster = self.data[self.data["cluster"] == cls]
            if len(cluster) < 2:
                continue

            # percent ratio of values are smaller than p_thres
            if self.if_norm_loss:
                print("using normed loss...")
                cluster["seq_loss_norm"] = cluster["seq_loss"]/cluster["masked_num"]*100
                loss_thres = self.calculate_cluster_p_threshold(ratios_dict[cls], cluster["seq_loss_norm"])
                selected_ids = cluster[cluster["seq_loss_norm"] > loss_thres]["seq_id"]
            else:
                print("using unnormed loss...")
                loss_thres = self.calculate_cluster_p_threshold(ratios_dict[cls], cluster["seq_loss"])
                selected_ids = cluster[cluster["seq_loss"] > loss_thres]["seq_id"]
            selection_ids += selected_ids.tolist()


        return np.array(selection_ids, dtype=np.int64)

    def get_outlier_clusters(self, consider_loss = False):
        score_thres = np.percentile( np.array(list(self.est_anomaly_ratio_dict.values())), self.outlier_cluster_thres)
        if_outlier_cluster_dict = {c: r > score_thres for c, r in self.est_anomaly_ratio_dict.items()}
        print(f"outlier cluster {if_outlier_cluster_dict}")

        #todo: to be compatible with tbird_V7_SPLIT, consider the cluster with large elbow point of domiance also as outlier cluster
        cluster_domiance_df = pd.DataFrame()
        cluster_domiance_df["cluster"] = list(self.cluster_dominance_dict.keys())
        cluster_domiance_df["dominance"] = list(self.cluster_dominance_dict.values())
        cluster_domiance_df = cluster_domiance_df.sort_values(by = ["dominance"], ascending= True)
        cluster_domiance_df = cluster_domiance_df.reset_index(drop=True)
        # find the elbow of the dominance
        kn = KneeLocator(cluster_domiance_df["cluster"], cluster_domiance_df["dominance"], curve='convex', direction='increasing')
        # kn.knee: the index of the elbow
        elbow_cluster = cluster_domiance_df.iloc[kn.knee]["cluster"]
        elbow_dom = cluster_domiance_df.iloc[kn.knee]["dominance"]
        large_dom_clusters = cluster_domiance_df[cluster_domiance_df["dominance"] >= elbow_dom]["cluster"]
        cluster_thres = self.outlier_cluster_thres/100 * len(cluster_domiance_df)
        # enable the outlier clusters
        if kn.knee +1 > cluster_thres:
            for c in large_dom_clusters:
                if_outlier_cluster_dict[c] = True

        return  if_outlier_cluster_dict


#
if __name__ == '__main__':

    data = pd.read_csv(f"../../../viz/spirit_admin_split/original_source_0.csv")
    # data = data.drop(columns=["cluster","center","eu_distance","cos_distance","first_score","sec_score","selected","outlier_cluster","cluster_anomaly_ratio"])
    # output_path = f"{dir}/seq_info_epoch_{e}_clusted.csv"
    random_state = 1234
    finer = pluto(data, if_norm_loss =True, if_process_outlier = True, remove_large_loss = True, random_state=random_state, if_select=False)

    selection = finer.run(depth=1, save_output=False, output_path=f"../../../viz/spirit_admin_split/original_source_0_pluto.csv")
    # selection.to_csv(f"../../../viz/tbird/5-7M_SPLIT/source_0_pluto_selection.csv")