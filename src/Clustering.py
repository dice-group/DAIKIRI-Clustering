from pandas.io import pickle
from sklearn.metrics.pairwise import pairwise_distances

import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from src.DataLoader import DataLoader

import os

# In this step, we group entities with similar properites (i.e., based on their embedding representations) into clusters.
# Each group should have similar entities --> similar types.

# We employ a density-based clustering (hdbscan) to detect entities cluster based on their density in the embedding space.
# We use the implementation of hdbscan clustering library. For more information/install,
# please check https://hdbscan.readthedocs.io/en/latest/index.html


class Density_Clustering:

    def __init__(self, data, config):
        """ """
        self.hdbscan_clusters = hdbscan.HDBSCAN(algorithm='best', alpha=config['alpha'], metric=config['metric'], cluster_selection_method=config['cluster_selection_method'],
                                                min_samples=config['min_samples'], min_cluster_size=config['min_cluster_size'], core_dist_n_jobs=-1, allow_single_cluster=config['allow_single_cluster'],
                                                cluster_selection_epsilon=config['cluster_selection_epsilon']).labels_

        self.distance_matrix = pairwise_distances(data, metric='cosine')

    def save_Output(self, path="output"):
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(self.hdbscan_clusters, open(
            path+'/hdbscanClusters_Output.pkl', 'wb'))
        return self.hdbscan_clusters


class Centroid_Clustering: 

    def __init__(self, data, config):
        kmeans = KMeans(n_clusters=config['n_clusters'] ).fit(data)

        self.Kmeans_clusters= self.kmeans.predict(data)

    def save_Output(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        pickle.dump(self.Kmeans_clusters, open(
            'output/KmeansClusters_Output.pkl', 'wb'))

class Agglomerative_Clustering:

    def __init__(self, data, config):

        self.aggClustering = AgglomerativeClustering(n_clusters=config['n_clusters'])
        self.agg_Clusters= self.aggClustering.fit_predict(data)

    def save_Output(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        pickle.dump(self.agg_Clusters, open('output/aggClusters_Output.pkl', 'wb'))



       


