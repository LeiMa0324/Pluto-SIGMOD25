import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class selection_measure():
    def __init__(self, data, selection_data=None, marks = None, ori_domiance = None, sel_dominance = None):
        self.data = data
        self.selection_data = selection_data

        self.ori_sizes = 0
        self.ori_anomalies = 0
        self.ori_anomaly_ratios = 0.0
        self.ori_cluster_sizes = {}
        self.ori_cluster_anomalies = {}
        self.ori_cluster_anomaly_ratios = {}
        self.ori_dominance = ori_domiance
        self.ori_loss = None

        self.sel_sizes = 0
        self.sel_anomalies = 0
        self.sel_unique_anomalies = 0
        self.sel_anomaly_ratios = 0.0
        self.sel_cluster_sizes = {}
        self.sel_cluster_anomalies = {}
        self.sel_cluster_anomaly_ratios = {}
        self.sel_dominance = sel_dominance

        self.marks = marks


    def measure(self):
        self.ori_cluster_sizes, self.ori_cluster_anomalies, self.ori_cluster_anomaly_ratios,\
        self.ori_anomalies, self.ori_sizes, self.ori_anomaly_ratios, self.ori_cluster_loss, self.ori_cluster_normal_loss,\
            self.ori_cluster_abnormal_loss = self.cluster_measure(self.data)
        if self.selection_data is not None:
            self.sel_cluster_sizes, self.sel_cluster_anomalies, self.sel_cluster_anomaly_ratios, \
                    self.sel_anomalies, self.sel_sizes, self.sel_anomaly_ratios , self.sel_cluster_loss \
                , self.sel_cluster_normal_loss, \
            self.sel_cluster_abnormal_loss            = self.cluster_measure(self.selection_data)


    def cluster_measure(self, data):
        cluster_sizes = {}
        cluster_anomalies={}
        cluster_anomaly_ratios ={}
        cluster_loss = {}
        cluster_normal_loss = {}
        cluster_abnormal_loss = {}
        total_anomalies = len(data[data["seq_label"]==1])
        total_size = len(data)
        total_anomaly_ratio = float(total_anomalies)/float(total_size) if total_size>0 else 0
        lef_clusters = len(data["cluster"].unique())

        for c in data["cluster"].unique():
            cluster = data[data["cluster"]==c]
            size = len(cluster)
            normal = cluster[cluster["seq_label"]==0]
            abnormal = cluster[cluster["seq_label"]==1]
            anomaly_num = len(cluster[cluster["seq_label"]==1])
            anomaly_ratio = float(anomaly_num)/float(size) if size > 0 else 0.0
            cluster_sizes[c] = size
            cluster_anomalies[c] = anomaly_num
            cluster_anomaly_ratios[c] = round(anomaly_ratio,3)
            if "seq_loss" in cluster.columns:
                cluster_loss[c] = cluster["seq_loss"].mean()
                cluster_normal_loss[c] = round(normal["seq_loss"].mean(),1) if len(normal)> 0 else 0
                cluster_abnormal_loss[c] = round(abnormal["seq_loss"].mean(), 1) if len(
                    abnormal) > 0 else 0

        return cluster_sizes, cluster_anomalies, cluster_anomaly_ratios, total_anomalies, total_size, total_anomaly_ratio, cluster_loss, cluster_normal_loss, cluster_abnormal_loss

    def print_measures(self, level = 'summary'):
        if level == 'all' or level =='cluster':
            print("********* Printing cluster measurements... *********")
            for c in range(0, len(self.ori_cluster_sizes.keys())):

                print(f"*cluster {c}" if self.marks is not None and c in self.marks.keys() and self.marks[c] else f"cluster {c}")
                if c not in self.ori_cluster_sizes.keys():
                    continue
                ori_message = f"ori cluster size: {self.ori_cluster_sizes[c]}, ori anomaly num :{self.ori_cluster_anomalies[c]}, ori anomaly ratio:{round(self.ori_cluster_anomaly_ratios[c], 3)}"
                if self.ori_dominance is not None:
                    ori_message += f", dominance: {round(self.ori_dominance[c], 1)}"
                if len(self.ori_cluster_loss)>0:
                    ori_message += f", mean loss: {round(self.ori_cluster_loss[c], 1)}"
                if len(self.ori_cluster_normal_loss) > 0:
                    ori_message += f", normal mean loss: {round(self.ori_cluster_normal_loss[c], 1)}"
                if len(self.ori_cluster_abnormal_loss) > 0:
                    ori_message += f", abnormal mean loss: {round(self.ori_cluster_abnormal_loss[c], 1)}"

                sel_message = f"sel cluster size: {self.sel_cluster_sizes[c] if c in self.sel_cluster_sizes.keys() else 0}," \
                              f" sel anomaly num :{self.sel_cluster_anomalies[c] if c in self.sel_cluster_anomalies.keys() else 0}," \
                              f" sel anomaly ratio:{round(self.sel_cluster_anomaly_ratios[c],3) if c in self.sel_cluster_anomaly_ratios.keys() else 0}"
                if self.sel_dominance is not None:
                    sel_message += f", dominance: {round(self.sel_dominance[c], 1)}"
                if len(self.sel_cluster_loss) > 0 and c in self.sel_cluster_loss.keys():
                    sel_message += f", mean loss: {round(self.sel_cluster_loss[c], 1)}"
                if len(self.sel_cluster_normal_loss) > 0 and c in self.sel_cluster_normal_loss.keys():
                    sel_message += f", normal mean loss: {round(self.sel_cluster_normal_loss[c], 1)}"
                if len(self.sel_cluster_abnormal_loss) > 0  and c in self.sel_cluster_abnormal_loss.keys():
                    sel_message += f", abnormal mean loss: {round(self.sel_cluster_abnormal_loss[c], 1)}"
                print(ori_message)
                print(sel_message)

        print("================")
        print(f"ori size: {self.ori_sizes}, ori anomaly ratio: {round(self.ori_anomaly_ratios,3)}, ori anomaly num: {self.ori_anomalies}")
        print(f"sel size: {self.sel_sizes}, sel anomaly ratio: {round(self.sel_anomaly_ratios,3)}, sel anomaly num: {self.sel_anomalies}")


    def plot_ori_cluster_statistics(self, title):
        abnor_nums = []
        nor_nums = []
        cluster_name = []
        for c in self.ori_cluster_sizes.keys():
            norm_num = self.ori_cluster_sizes[c] -self.ori_cluster_anomalies[c]
            ab_num = self.ori_cluster_anomalies[c]
            abnor_nums.append(ab_num)
            nor_nums.append(norm_num)
            cluster_name.append(c)
        x = np.arange(len(nor_nums))
        abnor_nums = np.array(abnor_nums)
        nor_nums = np.array(nor_nums)

        plt.bar(x, abnor_nums, width=0.2, color = 'darkorange')
        plt.bar(x, nor_nums, bottom=abnor_nums, width=0.2, color='steelblue')
        plt.xticks(cluster_name)
        plt.xlabel("cluster")
        plt.ylabel("# sequence")
        plt.title(title)
        plt.show()

    def ori_adjacent_matrix(self):
        seq_id_np = []
        adj_mat = np.zeros((len(self.data), len(self.data)))
        for c in self.data["cluster"].unique():
            cluster = self.data[self.data["cluster"]==c]
            seq_index = cluster.index.tolist()
            seq_id_np.extend(cluster["seq_id"].to_numpy().tolist())
            for i in seq_index:
                for j in seq_index:
                    adj_mat[i][j] = 1

        return adj_mat, seq_id_np
