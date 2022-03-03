import pandas as pd

from sklearn.metrics.pairwise import pairwise_distances

import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from data_loader import DataLoader

import os

# In this step, we group entities with similar properites (i.e., based on their embedding representations) into clusters.
# Each group should have similar entities --> similar types.

# We employ a density-based clustering (hdbscan) to detect entities cluster based on their density in the embedding space.
# We use the implementation of hdbscan clustering library. For more information/install,
# please check https://hdbscan.readthedocs.io/en/latest/index.html




class Model: 
    def __init__(self, model, data, config, output_path):
        
        if model=="hdbscan": 
            Density_Clustering(data, config, output_path)
        elif model =="kmeans":
            Centroid_Clustering(data, config, output_path)
        elif model=="agglomerative":
            Agglomerative_Clustering(data, config, output_path)
                  

class Density_Clustering:

    def __init__(self, data, config, output_path):
        
        self.data= data
        self.hdbscan = hdbscan.HDBSCAN(algorithm='best', alpha=config['alpha'], metric=config['metric'], cluster_selection_method=config['cluster_selection_method'],
                                                min_samples=config['min_samples'], min_cluster_size=config['min_cluster_size'], core_dist_n_jobs=-1, allow_single_cluster=config['allow_single_cluster'], cluster_selection_epsilon=config['cluster_selection_epsilon'])
        
        self.distance_matrix = pairwise_distances(self.data, metric='cosine')
        
        self.hdbscan_clusters=self.hdbscan.fit(self.distance_matrix).labels_        
        self.save_Output(path=output_path)

    
    
    def save_Output(self, path="../output"):
        if not os.path.exists(path):
            os.makedirs(path)
                        
        df_tmp = pd.DataFrame({'clusters_id': self.hdbscan_clusters}, index=self.data.index)
        df_tmp.index.name='entity-ID'
        df_tmp['clusters_id']= df_tmp['clusters_id'].apply(lambda x: "cluster-"+str(x))
        df_tmp.to_csv(path+"/hdbscan_Clusters.csv", header=True)

        return df_tmp


class Centroid_Clustering: 

    def __init__(self, data, config, output_path):
        
        self.data=data
        self.kmeans = KMeans(n_clusters=config['n_clusters'] ).fit(self.data)

        self.Kmeans_clusters= self.kmeans.predict(self.data)
        self.save_Output(path=output_path)

    def save_Output(self, path="../output"):
        if not os.path.exists("output"):
            os.makedirs("output")

        df_tmp = pd.DataFrame({'clusters_id': self.Kmeans_clusters}, index=self.data.index)
        df_tmp.index.name='entity-ID'
        df_tmp['clusters_id']= df_tmp['clusters_id'].apply(lambda x: "cluster-"+str(x))        
        df_tmp.to_csv(path+"/kmeans_Clusters.csv")
        
        return df_tmp

class Agglomerative_Clustering:

    def __init__(self, data, config, output_path):

        self.data=data
        self.aggClustering = AgglomerativeClustering(n_clusters=config['n_clusters'])
        self.agg_Clusters= self.aggClustering.fit_predict(self.data)
        self.save_Output(path=output_path)

    def save_Output(self, path="../output"):
        if not os.path.exists("output"):
            os.makedirs("output")

        df_tmp = pd.DataFrame({'clusters_id': self.agg_Clusters}, index=self.data.index)
        df_tmp.index.name='entity-ID'
        df_tmp['clusters_id']= df_tmp['clusters_id'].apply(lambda x: "cluster-"+str(x))
        df_tmp.to_csv(path+"/agglo_Clusters.csv")
        
        return df_tmp

       


