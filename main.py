import os
import pandas as pd
import numpy as np
import hdbscan
import argparse

from clustering_evaluation import ClusterPurity

    
def hdbscan_clustering(cluster_df):
    """
    param: Embeddings dataframe to be clustered
    return: a clustered dataframe
    """ 
    ##-- Cluster the data using HDBSCAN --### 
    clusterer = hdbscan.HDBSCAN(
        algorithm='best', 
        alpha=1.3,
        approx_min_span_tree=True,
        metric='euclidean',
        gen_min_span_tree=True,
        min_cluster_size=10000,
        min_samples=100,
        cluster_selection_epsilon= 0.5,
        core_dist_n_jobs=1,
        allow_single_cluster=False).fit(cluster_df)

    #-- Save clustering results as a new column in the original dataframe --#
    cluster_df['cluster_preds']=clusterer.labels_
    
    print ('Number of Clusters: ',len(set(clusterer.labels_)))

    return cluster_df


def evaluate_ClusteringPurity(cluster_preds, cluster_true):
    """
    param: 
        cluster_preds: predict cluster label
        cluster_true: the ground truth labels of clusters. Here, we use 'type' column from 
        the original dataset.
        
    return purity_score
    """

    evaluator=ClusterPurity()
    purity_score=evaluator.purity_score(y_true=cluster_true, y_pred=cluster_preds)

    return purity_score


def cluster_distriubions(clustered_df):
    
    print('Clusters Distributions: ')
    cluster_distributions=clustered_df.groupby('cluster_preds').size()
    for cluster, size in cluster_distributions.items():
        print (cluster, size)
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default='', nargs="?",
                        help="Path of input data e.g. ai4bd.csv or merged.csv")
    #input_data='/home/daikiri/DAIKIRI/src/Hamada/merged.csv' # use your own path
    
    parser.add_argument("--embedding_data", type=str, default='', nargs="?",
                        help="full path of embedding data e.g.Vectograph_Results/2020-08-08 01:00:07.899851/PYKE_50_embd.csv")
    #embedding_data='/home/daikiri/DAIKIRI/src/Hamada/Vectograph_Results/2020-08-08 01:00:07.899851/PYKE_50_embd.csv' 

    parser.add_argument("--config", type=str, default='', nargs="?",
                        help="full path of config file to read hyper-parameter values. e.g. param.txt")

    args = parser.parse_args()
    
    # -- load the input dataset--#
    input_df=pd.read_csv(args.input_data, low_memory=True)
    input_df.index = 'Event_' + input_df.index.astype(str)
    num_rows, num_cols = input_df.shape  # at max num_rows times num_cols columns.
    column_names = input_df.columns
    print ('input_data (ai4bd) loaded with shape', input_df.shape) 
    
    #-- load the embeddings data --#
    embedding_input=pd.read_csv(args.embedding_data, index_col=0, low_memory=True)
    #consider only events embedding (Event_id)
    embedding_index=embedding_input.index.tolist()
    prefix = 'Event_'
    event_ids=list(filter(lambda x: x.startswith(prefix), embedding_index))
    events_df=embedding_input.loc[event_ids]    
    print ('embedding_data loaded with shape', events_df.shape )


    #-- Clustering using HDBSCAN --#
    clustered_df=hdbscan_clustering(events_df)

    clustered_df.to_csv('./clustered_data.csv')

    #-- Evaluate the clustering performance --#
    cluster_preds= events_df['cluster_preds'].tolist()
    cluster_true= input_df['type'].tolist()

    purity_score=evaluate_ClusteringPurity(cluster_preds, cluster_true)
    print('Clustering Purity Score: ', purity_score)
        
    # -- Print Cluster Distributions --#
    cluster_distriubions(clustered_df)
    