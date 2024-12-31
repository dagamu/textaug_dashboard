import numpy as np
import pandas as pd
import os
import re
import ast

from skmultilearn.dataset import load_dataset, available_data_sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import datasets
import requests

class Formatter:
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        
    def format_text(self, text, format_):
        
        fn_dict = { "TEXT": lambda x: x } # [0]?
        if not format_ in fn_dict:
            print(f"{format_} format not supported")
            
        return [fn_dict[format_](value) for value in text]
    
    def format_labels(self, labels, format_, sep=',', labels_names=[]):
        
        if format_ == "BINARY":
            if len(labels_names) == 0:
                labels_names = [ f"L{i}" for i in range(labels.shape[1]) ]
        
        fn_dict = { 
                    "LIST": lambda y: y,
                    "SEP": lambda y: y.split(sep),
                    "BINARY_LIST": lambda y: labels_names[np.array(y, dtype=bool)],
                    "LITERAL": lambda y: ast.literal_eval(y)
                }
        
        if not format_ in fn_dict:
            print(f"{format_} format not supported")
            
        return map(fn_dict[format_], labels)
        return self.mlb.fit_transform(y_list)

class Dataset:
    source = "UNKNOWN"
    name = "Unknown"
    
    loaded = False
    preprocessed = False
    labels_provided = False
    
    columns = None
    X_aug = []
    y_aug = []
    
    """
    X_format = [ TEXT | FREQ ] -> TODO: TEXT_COLUMNS
    y_format = [ LIST | BINARY | BINARY LIST | SEP | LITERAL ]
    """
    format_sep = ','
    
    default_params = {  
                        "key": None,
                        "key_test": None,
                        "text_column": None,
                        "labels_column": None,
                        "columns_indices": {"X":(0,1), "Y":(1,1)},
                        "make_split": False,
                        "random_seed": 42,
                        "X_format": "TEXT",
                        "y_format": "LIST",
                        "labels_names": []
                      }
    
    def __init__(self, **kwargs):
        self.preload(**kwargs)
        
        self.params = { **self.default_params, **kwargs } 
        self.key_train = self.params["key"]
        self.key_test = self.params["key_test"]
        self.columns_indices = self.params["columns_indices"]
        self.make_split = self.params["make_split"]
        self.random_seed = self.params["random_seed"]
        
        self.name = self.get_name()
        self.setup(**kwargs)
        self.formatter = Formatter()
        self.labels_provided = "labels_names" in kwargs
        
    def preload(self, **kwargs):
        pass
    
    def setup(self, **kwargs):
        pass
    
    def full_train_data(self, mask=[]):
        if len(mask):
            X = [ x_ for i, x_ in enumerate(self.X_train) if mask[i] ] 
            y = [ y_ for i, y_ in enumerate(self.y_train) if mask[i] ] 
        else:
            X = list(self.X_train)
            y = list(self.y_train)
        
        if len(self.X_aug) and len(self.y_aug):
            X += self.X_aug
            y += self.y_aug
            
        return X, y
        
    def get_name(self):
        return self.key_train
    
    def update_data(self, X_train, X_test, y_train, y_test):
        self.X_train = self.formatter.format_text(X_train, self.params["X_format"])
        self.X_test  = self.formatter.format_text(X_test, self.params["X_format"])
    
        l = self.labels if self.labels_provided else []
        self.y_train = self.formatter.format_labels(y_train, self.params["y_format"], self.format_sep, labels_names=l )
        self.y_test  = self.formatter.format_labels(y_test, self.params["y_format"], self.format_sep, labels_names=l )
    
    def column_info_to_range(self, columns_indices, columns=[]):
        
        if self.params["text_column"]:
            self.columns_indices['X'] = (columns.index(self.params["text_column"]), 1)
            
        if self.params["labels_column"]:
            self.columns_indices['Y'] = (columns.index(self.params["labels_column"]), 1)
        
        X_range = range( columns_indices['X'][0], columns_indices['X'][0] + columns_indices['X'][1] )
        y_range = range( columns_indices['Y'][0], columns_indices['Y'][0] + columns_indices['Y'][1] )
        return X_range, y_range
    
    def get_pandas_data(self):
        df = pd.read_csv(self.key_train)
        columns = df.columns
        
        X_range, y_range = self.column_info_to_range(self.columns_indices, list(columns) )
        X_cols = columns[X_range]
        y_cols = columns[y_range]
        
        
        if self.make_split:
            X_features = df[X_cols].sum(axis=1)
            y_features = df[y_cols].sum(axis=1)
            return train_test_split(X_features.to_list(), y_features.to_list(), test_size=0.3)
            
        else:
            df_train = df
            df_test = pd.read_csv(self.test_key)
            X_train = df_train[X_cols].sum(axis=1)
            X_test = df_test[X_cols].sum(axis=1)
            y_train = df_train[y_cols].sum(axis=1)
            y_test = df_train[y_cols].sum(axis=1)
            return X_train.to_list(), X_test.to_list(), y_train.to_list(), y_test.to_list()
    
class ThirdFileDS(Dataset):
    
    source = "UPLOADED_FILE"
        
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
    
    def preload(self, **_):
        self.default_params["make_split"] = True
        
    def setup(self, **_):
        self.key_train = f'data/{self.key_train}'
        if self.key_test:
            self.key_test = f'data/{self.key_test}'
    
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
        self.labels = np.array(self.params["labels_names"]) if self.params["labels_names"] != None else [f"L{i+1}" for i in range(self.y_train[0])]
        self.loaded = True

class HuggingFaceDS(Dataset):
    
    source = "HUGGINFACE_DATASETS"
    df_obj = None
    
    def preload(self, **kwargs):
        self.split = kwargs["split"] if "split" in kwargs else "train"
    
    def dataset_exists(self):
        res = requests.get(f"https://huggingface.co/datasets/{self.key}")
        return res.status_code == 200

    def get_columns(self):
        if type(self.columns) == type(None):
            header = datasets.load_dataset(self.key_train, split="train[:1]", trust_remote_code=True).to_pandas()
            self.columns = header.columns
        return self.columns
    
    def get_data(self):
        self.loaded = True
        
        columns = self.get_columns()
        X_range, y_range = self.column_info_to_range(self.columns_indices, list(columns) )
        self.labels = np.array(self.params["labels_names"]) if self.params["labels_names"] != None else np.take(columns, y_range)
        
        X_cols = columns[X_range]
        y_cols = columns[y_range]
        
        if self.make_split:
            df = datasets.load_dataset(self.key_train, split=self.split, trust_remote_code=True).to_pandas()
            split = train_test_split(df[X_cols].sum(axis=1).to_list(), df[y_cols].sum(axis=1).to_list(), test_size=0.3)
            self.update_data(*split)
            
        else:
            ds = datasets.load_dataset(self.key_train, trust_remote_code=True)
            df_train = ds["train"].to_pandas()  
            df_test = ds["test"].to_pandas()
            self.update_data( df_train[X_cols].sum(axis=1).to_list(),
                              df_test[ X_cols].sum(axis=1).to_list(),
                              df_train[y_cols].sum(axis=1).to_list(),
                              df_test[ y_cols].sum(axis=1).to_list())


class SKMultilearnDS(Dataset):
    
    source = "SCIKIT_MULTILEARN"
    
    def setup(self, **kwargs):
        self.key = self.key_train
        self.variant = self.key_test if self.key_test else "train"
        self.default_params["X_format"] = "FREQ"
        self.default_params["y_format"] = "BINARY"
    
        
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
