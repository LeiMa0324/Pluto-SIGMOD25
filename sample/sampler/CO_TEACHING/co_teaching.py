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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


warnings.filterwarnings('ignore')

class coteaching():
    def __init__(self,  anomaly_percent, epochs, method):

        self.noisy_rate = 0.2
        self.epochs = epochs
        # how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.
        self.num_gradual = 20
        # exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.
        self.exponent = 1
        self.schedule_rate()
        self.method = method

    def schedule_rate(self):
        # wandb.config.update({"noisy_rate": self.noisy_rate,
        #                      "num_gradual": self.num_gradual,
        #                      "exponent": self.exponent})

        self.rate_schedule = np.ones(self.epochs) * self.noisy_rate
        self.rate_schedule[:self.num_gradual] = np.linspace(0, self.noisy_rate ** self.exponent, self.num_gradual)

    def run(self, data, epoch):
        self.data = data
        model1_selection, model2_selection = self.loss_coteaching( epoch)
        self.data["selection_1"] = self.data["seq_id"].isin(model1_selection["seq_id"])
        self.data["selection_2"] = self.data["seq_id"].isin(model2_selection["seq_id"])

        anomaly_num_model1 = len(model1_selection[model1_selection["seq_label"]==1])
        anomaly_num_model2 = len(model2_selection[model2_selection["seq_label"] == 1])

        num_model1 = len(model1_selection)
        num_model2 = len(model2_selection)


        return anomaly_num_model1, anomaly_num_model2, \
               num_model1, num_model2, \
               torch.from_numpy(self.data["seq_id"].isin(model1_selection["seq_id"]).to_numpy().astype(int)), \
        torch.from_numpy(self.data["seq_id"].isin(model2_selection["seq_id"]).to_numpy().astype(int)), self.data


    # Loss functions
    def loss_coteaching(self,  epoch):

        if self.method == 'coteaching':
            sort_data_loss1 = self.data.sort_values(by=['loss_1'])
            sort_data_loss2 = self.data.sort_values(by=['loss_2'])
        elif self.method == 'coteaching_norm':
            self.data["loss_1_norm"] = self.data['loss_1']/self.data["mask_num"] * 100
            self.data["loss_2_norm"] = self.data['loss_2']/self.data["mask_num"] * 100
            sort_data_loss1 = self.data.sort_values(by=['loss_1_norm'])
            sort_data_loss2 = self.data.sort_values(by=['loss_2_norm'])
        else:
            raise NotImplementedError


        self.remember_rate = 1 - self.rate_schedule[epoch]
        self.num_remember = int(self.remember_rate * len(self.data))

        model2_selection = sort_data_loss1[:self.num_remember] # the samples with small loss of model 1
        model1_selection = sort_data_loss2[:self.num_remember] # the samples with small loss of model 2


        print(f"number remember: {self.num_remember}, model 1 selection size: {len(model1_selection)}, model 2 selection size: {len(model2_selection)}")


        return model1_selection, model2_selection


