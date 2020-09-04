# DAIKIRI-Clustering
* Create a first prototype for AP3.1 using HDBSCAN algorithm. 
* The clustering performance is measure using cluster_purity metric. 

---
## Clustering Demo using HDBSCAN

The module takes two inputs file: original data (e.g `ai4bd.csv`) and the embeddings data generated from the previous module, it can be found fore example in `Vectograph_Results/../PYKE_50_embd.csv` 

The clustering_demo (in main.py) uses hdbscan with tuned hyperparameters to cluster the events data based on their embeddings. 

Further, we use the `cluster_purity` as a main mertic to evaluate the clsutering performance based on `event_type`. The results can be shown in `Cluster Purity Score`. In addition, we use `cluster_validy_index` implement in `hdbscan` to compute the density based cluster validity index for the clustering specified by labels. 

Finally, the clustering_demo shows the clusters distributions and ratio of outliers.

---
## Contact

If you have any further questions/suggestions, please contact `hamada.zahera@upb.de`

