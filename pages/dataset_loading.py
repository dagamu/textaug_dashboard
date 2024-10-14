import pandas as pd
import importlib
import requests 
import os

from utils import LOADING_PRESETS

import streamlit as st
from menu import menu

# TODO: Make option to load binarized labels 

class FileTab:
    def __init__(self):
        self.name = "FileUpload"
        self.filepath = None
        
    def get_data(self):
        filename = self.filepath.name
        cols_to_read = set(self.cols) - set(self.cols_to_drop)
        return pd.read_csv(self.filepath, usecols=list(cols_to_read)), filename
    
    def get_columns(self):
        if self.filepath == None:
            return []
        self.cols = pd.read_csv(self.filepath, nrows=0).columns
        return self.cols
    
    def render(self, st):
        st.subheader("Load a CSV file")
        self.filepath = st.file_uploader( "Choose a CSV file", accept_multiple_files=False )

class DataFolderTab:
    def __init__(self):
        self.name = "DataFolder"
        self.filename = None
        
    def get_data(self):
        if not "cols" in self.__dict__:
            self.cols = self.get_columns()
            
        cols_to_read = set(self.cols) - set(self.cols_to_drop)
        return pd.read_csv(f"data/{self.filename}", usecols=list(cols_to_read) ), self.filename
    
    def get_columns(self):
        if self.filename == None:
            return []
        
        if self.filename in LOADING_PRESETS["data_folder"]:
            self.cols_to_drop = LOADING_PRESETS["data_folder"][self.filename]["cols_to_drop"]
            
        self.cols = pd.read_csv(f"data/{self.filename}", nrows=0).columns
        return self.cols
    
    def render(self, st):
        st.subheader("Select from data folder")
        
        file_options = [ file for file in os.listdir("data") if file.endswith(".csv") ]
        self.filename = st.selectbox("CSV files in /data folder", file_options, index=None )

class HuggingFaceTab:
    def __init__(self):
        self.name = "Hugging Face"
        self.df_key = None
        self.module_ready = False
        
    def dataset_exists(self, key):
        if not key:
            return False
        res = requests.get(f"https://huggingface.co/datasets/{key}")
        return res.status_code == 200
        
    def import_module(self):
        if self.module_ready:
            return 
        
        self.load_dataset = importlib.import_module("datasets").load_dataset
        self.module_ready = True
        
    def get_data(self):
        if not self.module_ready:
            self.import_module()
            
        df_name = self.df_key.split('/')[1]
        if not self.dataset_exists(self.df_key):
            return {}, -1
        
        df = self.load_dataset(self.df_key, split=self.df_split, trust_remote_code=True).to_pandas()
        if self.cols_to_drop:
            df = df.drop(self.cols_to_drop)
            
        return df, df_name
    
    def get_columns(self):
        if self.df_key == None or not self.dataset_exists(self.df_key):
            return []
        
        if not self.module_ready:
            self.import_module()
        
        return self.load_dataset( self.df_key, split="train[:1]" ).column_names
    
    def render(self, st):
        st.subheader("Load a dataset from Hugging Face Datasets Module") 
        st.info("trust_remote_code is activated", icon="ℹ")
        
        split_options = ["train", "validation", "test", "train+test", "train+test+validation"]
        self.df_split = st.selectbox("Dataset Split", split_options)
        self.df_key = st.text_input("Dataset Name", placeholder="user/dataset", on_change=self.import_module )
        
        st.text("Common Examples: qanastek/HoC")


def UploadBtn(st, method, set_df):
    if st.button("Upload", type="primary", key=method.name):
        st.toast('Loading Dataset')
        df, df_name = method.get_data()
        
        if df_name == -1:
            st.error("Dataset doesn't exists")
            return 
        
        set_df(df, df_name, method.text_col, method.labels_col )
        st.toast('Done!')
         
            
def column_selection(method):
    columns = method.get_columns()
    if len(columns):
        text_col, label_col, drop_col =  st.columns([1, 1, 1.8])
        method.text_col = text_col.selectbox("Select Text Column", columns, index=0 )
        method.labels_col = label_col.selectbox("Select Labels Column", columns, index= 1 if len(columns) > 1 else None  )
        
        drop_placeholder = []
        if "cols_to_drop" in method.__dict__.keys():
            drop_placeholder = method.cols_to_drop
        
        method.cols_to_drop = drop_col.multiselect("Select Columns to Drop", columns, drop_placeholder, placeholder="None")
    

def RenderPage(st, set_df):
    load_methods =  {
        "Upload file":  FileTab,
        "Data Folder":  DataFolderTab,
        "Hugging Face": HuggingFaceTab,
    } 
    
    for method, tab in zip( load_methods.values(), st.tabs(load_methods.keys()) ):
        with tab:
            load_method = method()
            load_method.render(st)
            column_selection(load_method)
            UploadBtn( st, load_method, set_df )

def update_session_dataset( df, df_name, text_col, labels_col ):
    st.session_state["df"] = df
    st.session_state["df_name"] = df_name
    st.session_state["text_col"] = text_col
    st.session_state["labels_column"] = labels_col
    st.session_state["labels_data"] = {}
    
    if "prev_dataset_dropped" in st.session_state:
        del st.session_state["prev_dataset_dropped"]

def DatasetPage():
    
    st.title("Load the Dataset")

    if "df" in st.session_state:
        st.markdown(f"**Selected Dataset: {st.session_state['df_name']}**" )
        
    RenderPage(st, update_session_dataset)
    
    
if __name__ == "__main__": 
    DatasetPage()
    menu()