import pandas as pd
import hdbscan
import time
from clustering_evaluation import ClusterPurity

#-- compute runtime at the beginning--#
start_time = time.time()

#path=
def hdbscan_clustering(path):
    """
    param: path of embeddings dataframe to be clustered
    return: clustered dataframe
    """
    cluster_df=pd.read_csv(path, index_col=0)
    print ('Data loaded with shape', cluster_df.shape )

    ##-- Cluster the data using HDBSCAN --### -- Consider Hyperparameter tuning later --#
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.3, approx_min_span_tree=True, metric='euclidean',
     gen_min_span_tree=True, min_cluster_size=1000, min_samples=100, cluster_selection_method='leaf', allow_single_cluster=False).fit(cluster_input)

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

    path='' # e.g. Vectograph_Results/2020-08-08 01:00:07.899851/PYKE_50_embd.csv
    clustered_df=hdbscan_clustering(path)

    purity_score=evaluate_ClusteringPurity(clustered_df)
    print(purity_score)

