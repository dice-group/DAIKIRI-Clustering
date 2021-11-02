import pandas as pd
import json

class DataLoader: 
    def __init__(self, data_path, config_path) -> None:
        #inputPath: path of the embedding file (CSV), generated from the embedding WP 2
        self.config_path=config_path
        self.data_df= pd.read_csv(data_path, header=0,index_col=0)
        print ('data shape', self.data_df.shape)
        
  
    def get_Configuration(self): 

        with open(self.config_path) as config_file:
            data = json.load(config_file)

        return data

