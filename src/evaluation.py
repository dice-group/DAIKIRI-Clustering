import numpy as np
from sklearn import metrics

class ClusterPurity:

    def __init__(self):
        """
        param:
        return
        """

    def purity_score(self, y_true, y_pred):
        """
        param: y_true: the ground_truth labels of clusters. 
               y_pred: the predicted cluster labels.
        return: the purity score of clustering
        """
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)