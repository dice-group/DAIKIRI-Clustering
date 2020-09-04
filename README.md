# DAIKIRI-Clustering
* Create a first prototype for AP3.1 using HDBSCAN algorithm. 
* The clustering performance is measure using cluster_purity metric. 

---
## Clustering Demo using HDBSCAN

### Input Data:
- The module takes two inputs file: original data (e.g `ai4bd.csv`).
- The embeddings data generated from the previous module, it can be found fore example in `Vectograph_Results/../PYKE_50_embd.csv` 

The clustering_demo (in main.py) uses hdbscan with tuned hyperparameters to cluster the events data based on their embeddings. 
### Tuning Hyperparameters
- `min_cluster_size` is set to 10000 samples per cluster
- `min_samples` is set to 100 to consider a core point in HDBSCAN
- `cluster_selection_epsilon` is set to 0.5 to merge small clusters.

Other parameters can be shown here:
```python
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.3, approx_min_span_tree=True, metric='euclidean', gen_min_span_tree=True, min_cluster_size=10000, min_samples=100, cluster_selection_epsilon= 0.5, core_dist_n_jobs=1, allow_single_cluster=False).fit(cluster_df)
```
### Evaluation Metrics
- Further, we use the `cluster_purity` as a main mertic to evaluate the clsutering performance based on `event_type`. The results can be shown in `Cluster Purity Score`.
- In addition, we use [`cluster_validy_index`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.validity_index) implement in `hdbscan` to compute the density based cluster validity index for the clustering specified by labels. 

Finally, the clustering_demo shows the clusters distributions and ratio of outliers.

---
## Contact

If you have any further questions/suggestions, please contact `hamada.zahera@upb.de`

