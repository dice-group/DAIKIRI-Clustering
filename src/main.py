from data_loader import DataLoader
from clustering_models import Model

import argparse
  

def main():
          
    # creating an ArgumentParsert object
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embedding_file', default="../data/QMult_entity_embeddings.csv",help="Path of input file, csv embeddings", required=True)
    parser.add_argument('--model', help="specify the clustering model: hdbscan, kmeans or agglomerative", required=True)
    parser.add_argument('--config_Path', default="../configuration.json" ,help="Path of configuration file", required=True)
    parser.add_argument('--output_Path', default="../output", help="Path of output file, clustering results")
    
    args = parser.parse_args()

    loader= DataLoader(data_path=args.embedding_file, config_path=args.config_Path)
    config= loader.get_Configuration()

    #2) Data Clustering
    clusters=Model(model=args.model, data=loader.data_df, config= config, output_path=args.output_Path)


    

if __name__ == "__main__":
    main()
