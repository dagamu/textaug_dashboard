import os
import re
import pandas as pd
import numpy as np

from skmultilearn.dataset import load_dataset, available_data_sets
from sklearn.model_selection import train_test_split

import datasets
import requests

class Dataset:
    source = "UNKNOWN"
    name = "Unknown"
    loaded = False
    preprocessed = False
    columns = None
    
    X_aug = None
    y_aug = None
    
    """
    X_format = [ TEXT | FREQ ] -> TODO: TEXT_COLUMNS
    y_format = [ LIST | BINARY | SEP | LITERAL ]
    """
    format_sep = ','
    
    default_params = {  
                        "key": None,
                        "key_test": None,
                        "columns_indices": {"X":(0,1), "Y":(1,1)},
                        "make_split": False,
                        "random_seed": 42,
                      }
    
    def __init__(self, **kwargs):
        self.preload(**kwargs)
        
        params = { **self.default_params, **kwargs } 
        self.key_train = params["key"]
        self.key_test = params["key_test"]
        self.columns_indices = params["columns_indices"]
        self.make_split = params["make_split"]
        self.random_seed = params["random_seed"]
        
        self.name = self.get_name()
        self.setup(**kwargs)
        
    def preload(self, **kwargs):
        pass
    
    def setup(self, **kwargs):
        pass
    
    def full_train_data(self):
        X = np.ravel(self.X_train)
        y = self.y_train
        
        if np.all(self.X_aug != None) and len(self.y_aug):
            X = np.concat( (X, self.X_aug ))
            y = np.vstack( (y, self.y_aug ))
            
        return np.ravel(X), y
        
    def get_name(self):
        return self.key_train
    
    def update_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
    
    def column_info_to_range(self, columns_indices):
        X_range = range( columns_indices['X'][0], columns_indices['X'][0] + columns_indices['X'][1] )
        y_range = range( columns_indices['Y'][0], columns_indices['Y'][0] + columns_indices['Y'][1] )
        return X_range, y_range
    
    def get_pandas_data(self):
        df = pd.read_csv(self.key_train)
        columns = df.columns
        
        X_range, y_range = self.column_info_to_range(self.columns_indices)
        
        X_cols = columns[X_range]
        y_cols = columns[y_range]
        
        if self.make_split:
            return train_test_split(df[X_cols].values, df[y_cols].values, test_size=0.3)
            
        else:
            df_train = df
            df_test = pd.read_csv(self.test_key)
            return df_train[X_cols].values, df_test[X_cols].values, df_train[y_cols].values, df_test[y_cols].values
    
class ThirdFileDS(Dataset):
    
    source = "UPLOADED_FILE"
    X_format = "TEXT"
    y_format = "LIST"
        
    def preload(self, **kwargs):
        self.file = kwargs["key_train"]
        self.default_params["make_split"] = True
    
    def get_name(self):
        if self.name == "Unknown":
            m = re.search(r"[^/\\]+(?=\.[^/\\.]+$)", self.file.name)
            return m.group() if m else self.file.name
        return self.name
    
    def dataset_exists(self):
        return self.file or self.loaded
    
    def get_columns(self):
        if len(self.columns) == 0:
            header = pd.read_csv(self.file, nrows=0)
            self.columns = header.columns
        return self.columns
    
    def get_data(self):
        split = self.get_pandas_data()
        self.update_data(*split)
        self.loaded = True
        
class DataFolderDS(Dataset):
    
    source = "DATA FOLDER"
    X_format = "TEXT"
    y_format = "LIST"
    
    def preload(self, **kwargs):
        self.default_params["make_split"] = True
        
    def setup(self, **kwargs):
        self.key_train = f'data/{self.key_train}.csv'
        if self.key_test:
            self.key_test = f'data/{self.key_test}.csv'
    
    def get_name(self):
        return self.key_train.split('.')[0]
    
    def dataset_exists(self, path):
        return os.path.isfile(path)
    
    def get_columns(self):
        header = pd.read_csv(self.key_train, nrows=0)
        self.columns = header.columns
        return self.columns
    
    def get_data(self):
        if self.loaded:
            return
        split = self.get_pandas_data()
        self.update_data(*split)
        self.loaded = True

class HuggingFaceDS(Dataset):
    
    source = "HUGGINFACE_DATASETS"
    X_format = "TEXT"
    y_format = "LIST"
    df_obj = None
    
    def preload(self, **kwargs):
        self.split = kwargs["split"] if "split" in kwargs else "train"
            
    
    def dataset_exists(self):
        res = requests.get(f"https://huggingface.co/datasets/{self.key}")
        return res.status_code == 200

    def get_columns(self):
        if self.columns == None:
            header = datasets.load_dataset(self.key_train, split="train[:1]", trust_remote_code=True).to_pandas()
            self.columns = header.columns
        return self.columns
    
    def get_data(self):
        self.loaded = True
        
        columns = self.get_columns()
        X_range, y_range = self.column_info_to_range(self.columns_indices)
        X_cols = columns[X_range]
        y_cols = columns[y_range]
        
        if self.make_split:
            df = datasets.load_dataset(self.key_train, split=self.split, trust_remote_code=True).to_pandas()
            split = train_test_split(df[X_cols], df[y_cols], test_size=0.3)
            self.update_data(*split)
            
        else:
            ds = datasets.load_dataset(self.key_train, trust_remote_code=True)
            df_train = ds["train"].to_pandas()  
            df_test = ds["test"].to_pandas()
            self.update_data( df_train[X_cols], df_test[X_cols], df_train[self.y_cols], df_test[self.y_cols] )


class SKMultilearnDS(Dataset):
    
    source = "SCIKIT_MULTILEARN"
    X_format = "FREQ"
    y_format = "BINARY"
    
    def setup(self, **kwargs):
        self.key = self.key_train
        self.variant = self.key_test if self.key_test else "train"
        
    @staticmethod
    def get_avaiable_datasets():
        datasets = {}
        for ds, variant in available_data_sets().keys():
            if ds in datasets.keys():
                if variant in datasets[ds]:
                    continue
                else:
                    datasets[ds].append(variant)
            else:
                datasets[ds] = [variant]
        return datasets 
    
    def dataset_exists(self):
        if self.key in self.aviable_datasets.keys():
            return self.variant in self.aviable_datasets[self.key]
        return False
    
    def get_columns(self):
        if not self.loaded:
            self.get_data()
        return self.vocabulary + self.labels
    
    def get_data(self):
        
        if self.make_split:
            X_features, y_features, term_names, label_names = load_dataset(self.key, self.variant)
            split = train_test_split(X_features.toarray(), y_features.toarray(), test_size=0.3)
            self.update_data(*split)
        else:
            X_train, y_train, term_names, label_names = load_dataset(self.key, "train")
            X_test, y_test, _, _ = load_dataset(self.key, "test")
            split = [ X_train, X_test, y_train, y_test ]
            self.update_data(*[ item.toarray() for item in split ])
            
        self.vocabulary = [ name for name, _ in term_names ]
        self.labels = [ name for name, _ in label_names ]
        
        self.loaded = True
        

class DatasetManager:
    def __init__(self):
        self.items = []
    
    def load_dataset(self, source_label, params):
        sources_dict = { "DF": DataFolderDS, "TF": ThirdFileDS, "HF": HuggingFaceDS, "SKM": SKMultilearnDS }
        sources_alias = { "DATA_FOLDER": "DF", "FILE": "TF", "HUGGING_FACE": "HF" }
        for item in self.items:
            if item.key_train == params["key"]:
                return 
            
        if source_label in sources_dict:
            self.items.append( sources_dict[source_label](**params) )
        elif source_label in sources_alias:
            source = sources_alias[source_label]
            self.items.append( sources_dict[source](**params) )
        
    def remove(self, dataset):
        self.items = [ item for item in self.items if item != dataset ]
