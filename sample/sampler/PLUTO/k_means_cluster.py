import numpy as np
import pandas as pd
import ast

from sklearn.cluster import KMeans #K-Means Clustering


class KMeans_Cluster():
    def __init__(self,cluster_num, if_save=False, output_path=None):

        self.cluster_num = cluster_num
        self.if_save = if_save
        self.output_path = output_path


    def cluster(self, data, random_state= 1234):
        # data process
        X = data["embedding"].to_numpy()
        X_features = []
        for x in X:
            if isinstance(x, str):
                x = ast.literal_eval(x)
                x = np.array(x)
            X_features.append(x)
        X_features = np.array(X_features)

        sse = 0
        kmeans = KMeans(n_clusters= self.cluster_num, random_state= random_state).fit(X_features)
        # get the centroids of clusters
        centroids = kmeans.cluster_centers_
        # predict the cluster of each x
        pred_clusters = kmeans.predict(X_features)
        cluster_data = data
        cluster_data["cluster"] = pred_clusters
        cluster_data["center"] = cluster_data["cluster"].map(lambda c: centroids[c])
        eu_distances = []
        cos_distances = []
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(X_features)):

            curr_center = centroids[pred_clusters[i]]
            point = np.array(X_features[i])
            temp = point - curr_center
            # euclidean distance
            eu_dist = np.sqrt( np.dot(temp.T, temp))
            eu_distances.append(eu_dist)
            sse += eu_dist
            # cos distance
            cos_dist = 1 - np.dot(point, curr_center) / (
                    np.linalg.norm(point) * np.linalg.norm(curr_center))
            if cos_dist<0:
                cos_dist = 0
            cos_distances.append(cos_dist)

        cluster_data["eu_distance"] = np.array(eu_distances)
        cluster_data["cos_distance"] = np.array(cos_distances)
        # cluster_data["cluster_density"] = self.calculate_cluster_density(cluster_data)
        if self.if_save:
            cluster_data.to_csv(self.output_path+f"embedding_{self.cluster_num}_cluster_{iter}_iter.csv")

        return cluster_data

    def calculate_cluster_density(self, data):

        densities = []
        normal_samples = []
        abnormal_samples = []
        for c in range(0, self.cluster_num):
            cluster_data = data[data["cluster"]==c]
            density = len(cluster_data)/ cluster_data["eu_distance"].sum() if cluster_data["eu_distance"].sum()>0 else 0
            densities.append(density)
            norm = len(cluster_data[cluster_data["seq_label"]==0])
            normal_samples.append(norm)
            abnorm= len(cluster_data[cluster_data["seq_label"]==1])
            abnormal_samples.append(abnorm)

        return data["cluster"].map(lambda c: densities[c])
        # abnormal_samples = np.array(abnormal_samples)
        # normal_samples = np.array(normal_samples)
        #
        # sorted_cluster = np.argsort(denisties)
        # sorted_densities = np.sort(densities)
        #
        # sorted_ab = abnormal_samples[sorted_cluster]
        # sorted_nor = normal_samples[sorted_cluster]
        #
        #
        # plt.bar(np.arange(0, self.cluster_num)-0.2, sorted_densities*5000, width= 0.2, color = 'violet', label='density')
        # plt.bar(np.arange(0, self.cluster_num)+0.2, sorted_ab, width= 0.2,  color = 'darkorange', label='# abnormal')
        # plt.bar(np.arange(0, self.cluster_num)+0.2, sorted_nor, bottom= sorted_ab, width= 0.2,  color = 'steelblue', label='# normal')
        #
        #
        # plt.xticks(range(0, self.cluster_num),sorted_cluster)
        # plt.xlabel("cluster")
        # plt.ylabel("density")
        # plt.legend()
        # plt.title(f"iter {iter}, rate: {rate}")
        # plt.show()
        #
        # print(f"iter {iter}, density ranking: {sorted_cluster}")
        #




# dataset = "bgl"
# file = "files/bgl/bgl"
# train_file = "files/bgl/train/traintrain.csv"


#
# dataset = "bgl"
# k_map  = {"bgl": 20,
#           "tbird_V7_SPLIT": 25,
#           "HDFS": 20
#   }
#
# anomaly_ratios = [2, 5, 10, 15]
#
# #
# for rate in anomaly_ratios:
#     print(f"******* anomaly ratio: {rate}*******")
#     embedding_file = f"../../output/{dataset}/datasets/noisy_{rate}/embeddings/embedding.csv"
#     output_path = f"../../output/{dataset}/datasets/noisy_{rate}/embeddings/"
#     data = pd.read_csv(embedding_file)
#     train_file =  f"../../output/{dataset}/datasets/noisy_{rate}/traintrain.csv"
#     kmeans = KMeans_Cluster(cluster_num=k_map[dataset], iters=10, if_save=True, output_path=output_path)
#     cluster_iterations = kmeans.cluster(data)
#

