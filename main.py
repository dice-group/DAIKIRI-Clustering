import pandas as pd
import numpy as np
import hdbscan
from clustering_evaluation import ClusterPurity

def hdbscan_clustering(cluster_df):
    """
    param: path of embeddings dataframe to be clustered
    return: clustered dataframe
    """

    ##-- Cluster the data using HDBSCAN --### -- Consider Hyperparameter tuning later --#
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.3, approx_min_span_tree=True, metric='euclidean',
     gen_min_span_tree=True, min_cluster_size=1000, min_samples=100, cluster_selection_method='leaf', allow_single_cluster=False).fit(cluster_df)

    #-- Save clustering results as a new column in the original dataframe --#
    cluster_df['labels_preds']=clusterer.labels_
    
    print ('Number of Clusters',len(set(clusterer.labels_)))

    return cluster_df


def evaluate_ClusteringPurity(cluster_df):
    """
    param: cluster_df : clustered_dataframe
    return purity_score
    """

    labels_preds=cluster_df['labels_preds'].tolist()
    labels_true=cluster_df['labels_true'].tolist()

    evaluator=ClusterPurity()
    purity_score=evaluator.purity_score(y_true=labels_true, y_pred=labels_preds)

    return purity_score


if __name__ == "__main__":

    # -- load the clustering input --#
    path='/home/hzahera/Documents/SVN/Vectograph-develop/Vectograph_Results/2020-07-11 14:35:17.987772/PYKE_50_embd.csv' # e.g. Vectograph_Results/2020-08-08 01:00:07.899851/PYKE_50_embd.csv
    cluster_df=pd.read_csv(path, index_col=0)

    #- In case of sampling the input data, uncomment this code 
    cluster_df=cluster_df.sample(frac=0.3, replace=True, random_state=42)
    
    print ('Data loaded with shape', cluster_df.shape )

    # -- Add y_true as column with random values --#
    cluster_df['labels_true']=np.random.randint(-1, 880, cluster_df.shape[0]) # generate random cluster numbers range from -1 to total number of clusters

    clustered_df=hdbscan_clustering(cluster_df)

    purity_score=evaluate_ClusteringPurity(clustered_df)
    print('Cluster Purity Score: ', purity_score)

    print ('Done..')

