# DAIKIRI-Clustering

## Overview
This repository contains the implementation of clustering approaches used for work package 3 (Semantification). 
This project aims to semantifiy entities based on their embeddings representation in unsupervised learning cases (without labeled data). 

--- 
## Installation 
Install the requirements via ```pip install -r requirements.txt```

or manually using: 
* ```pip install -U scikit-learn```
* ```pip install hdbscan```

## How to run:

CommandLines: 

```cd src```

```python3 main.py --embedding_file="../data/QMult_entity_embeddings.csv" --model="hdbscan" --config_Path="../configuration.json" --output_Path="output"``` 

Parameters: 
* ```--embedding_file```: path of input file (csv format) contains the embeddings representation of data
* ```--model```: specify the clustering model: hdbscan, kmeans or agglomerative
* ```--config_Path```: path of configuration file
* ```--output_Path```: path of output file, clustering results
## Data: 

### *Input*: 
The input file represents the embedding representation of data (in CSV format) that was computed in the previous work package 2 (Embeddings), the file is loaded from input folder `data`.

### *Output*:

The output file is a CSV format, which contains the results of clustering (entity, cluster_ID), located into the generated folder 'output' 

---
## Contact
If you have any further questions or suggestions, please feel free to contact `hamada.zahera@upb.de`
