# DAIKIRI-Clustering

## Clustering Demo using HDBSCAN
To run a demo (in main.py) using hdbscan, type the following command line as follows:
<font size="-2">
```python
python3 main --input_data '/home/daikiri/DAIKIRI/src/Hamada/merged.csv' --embedding_data '/home/daikiri/DAIKIRI/src/Hamada/Vectograph_Results/2020-08-08 01:00:07.899851/PYKE_50_embd.csv'
```
</font>

### Commandline Arguments:
- `input_data`: path of input data e.g. ai4bd.csv or merged.csv 
- `embedding_data`: path of embedding data e.g.`Vectograph_Results/2020-08-08 01:00:07.899851/PYKE_50_embd.csv`

### Hyperparameters:

```python
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.3, approx_min_span_tree=True, metric='euclidean', gen_min_span_tree=True, min_cluster_size=10000, min_samples=100, cluster_selection_epsilon= 0.5, core_dist_n_jobs=1, allow_single_cluster=False).fit(cluster_df)
```

### Evaluation Metrics
- Further, we use the `cluster_purity` as a main mertic to evaluate the clsutering performance based on `event_type`. The results can be shown in `cluster purity score`.
- In addition, we use [`cluster_validy_index`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.validity_index) implement in `hdbscan` to compute the density based cluster validity index for the clustering specified by labels. 

Finally, the clustering_demo shows the clusters distributions and ratio of outliers.

---
### Evaluation Results

```
Data loaded with shape: (2974716, 41) 
Number of Clusters:  3
Clustering Purity Score:  0.9760810779919831
Clusters Distributions: 
outliers (-1), 4207
cluster (0), 114947
cluster (1), 2855562
```
---
### Clustering Output

The clustered data is saved into a csv file. To get each cluster samples, use: `clustered_df.groupby('cluster_preds')`

---
## Contact

If you have any further questions/suggestions, please contact `hamada.zahera@upb.de`
