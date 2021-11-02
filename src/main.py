from DataLoader import DataLoader
from Clustering import Density_Clustering


def main():

    #1) load embedding data and configuration
    data_path="./data/QMult_entity_embeddings.csv" 
    config_path="configuration.json"

    data= DataLoader(data_path, config_path)

    config= data.get_Configuration()

    #2) Data Clustering
    clusters=Density_Clustering(data, config)

    clusters_Output= clusters.save_Output()

    print (type(clusters_Output))

    #3) Labeling

if __name__ == "__main__":
    main()
