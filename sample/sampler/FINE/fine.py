import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import ast
import seaborn as sns
import warnings
from sample.selection_measure import selection_measure

warnings.filterwarnings('ignore')

class fine():
    def __init__(self, data):
        self.data = data
        self.features = None
        self.first_singular_vector_dict = None
        self.clean_labels = None
        self.parse_features()


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

    def get_singular_vector(self, vector_rank =0):
        '''
        To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
        features: hidden feature vectors of data (numpy)
        labels: correspoding label list
        '''

        singular_vector_dict = {}

        with tqdm(total=len(self.data["cluster"].unique())) as pbar:
            for c in self.data["cluster"].unique():
                _, _, v = np.linalg.svd(self.features[self.data["cluster"]==c])
                singular_vector_dict[c] = v[vector_rank] # the first singular vector
                pbar.update(1)

        return singular_vector_dict

    def run(self, p_thres = 0.5):

        self.first_singular_vector_dict = self.get_singular_vector()

        # calculate the score of each data point by the inner product of its feature with the singular vector
        self.data["first_score"] = self.get_score(self.first_singular_vector_dict)

        # selec # samples that equal to proporation 1- estimated anomaly ratio
        selection_id_first, first_large_mean_probs = self.fit_mixture( p_thres= p_thres)

        selection_first = self.data[self.data["seq_id"].isin(selection_id_first)]
        meas_1 = selection_measure(self.data, selection_first)
        meas_1.measure()
        meas_1.print_measures()

        return meas_1, selection_first



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


    def fit_mixture(self,  p_thres):
        '''
        Assume the distribution of scores: bimodal gaussian mixture model

        return noisy_0 labels
        that belongs to the noisy_0 cluster by fitting the score distribution to GMM
        '''

        selection_ids = []
        indices = []
        large_mean_probs = []


        for cls in range(0, len(self.data["cluster"].unique())):
            cluster = self.data[self.data["cluster"]==cls]
            scores = np.ravel( cluster["first_score"]).astype(float).reshape(-1, 1)

            gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)

            if len(scores)<2:
                continue
            gmm.fit(scores)
            prob = gmm.predict_proba(scores)
            index = cluster.index
            indices += index.tolist()
            prob = prob[:, gmm.means_.argmax()] # prob of larger mean distribution

            # percent ratio of values are smaller than p_thres
            if_clean = prob > p_thres
            selected_ids = cluster[if_clean]["seq_id"]
            selection_ids += selected_ids.tolist()

            large_mean_probs += prob.tolist()

        in_to_probs = pd.DataFrame()
        in_to_probs["index"] = np.array(indices)
        in_to_probs["large_mean_prob"] = np.array(large_mean_probs)
        in_to_probs.sort_values(by=["index"])


        return np.array(selection_ids, dtype=np.int64), \
               in_to_probs["large_mean_prob"].to_numpy()
