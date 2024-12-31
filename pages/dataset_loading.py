
import os
import streamlit as st
from menu import menu

from api.dataset import SKMultilearnDS

class SourceTab:
    
    key_train = None
    key_test = None
    columns_indices = {"X": [0], "y": [1]}
    make_split = False
    seed = None
    
    def get_params_alias(self):
        return {}
        
    def get_params(self):
        return { 
                    "key": self.key_train,
                    "columns_indices": {"X": (0, 1), "Y": (1, 1)},
                    "key_test": self.key_test,
                    "make_split": self.make_split,
                    "random_seed": self.seed,
                    **self.get_params_alias()
                }

class FileTab(SourceTab):
    
    name = "Load CSV"
    source_label = "TF"
    
    def render(self):
        st.subheader("Load a CSV file")
        
        train_col, test_col = st.columns(2)
        self.train_file = train_col.file_uploader("Choose a file for train data", accept_multiple_files=False )
        self.make_split = st.checkbox("Generate own train-test split", key="TF", value=False)
        
        if self.make_split:
            self.seed = test_col.text_input("Radom Seed for dataset split", value=42)
        else:
            self.test_file = test_col.file_uploader("Choose a file for test data", accept_multiple_files=False )

    def get_params_alias(self):
        return { "key_train": self.train_file, "key_test": self.test_file if not self.make_split else None }
            
class DataFolderTab(SourceTab):
    
    name = "Data Folder"
    source_label = "DF"
    
    def render(self):
        st.subheader("CSV files in /data folder")
        file_options = [ file for file in os.listdir("data") if file.endswith(".csv") ]
        train_col, test_col = st.columns(2)
        self.key_train = train_col.selectbox("Select train set", file_options, index=None )
        
        self.make_split = st.checkbox("Generate own train-test split", key="DF", value=True)
        if self.make_split:
            self.seed = test_col.text_input("Radom Seed for dataset split", key="DF_seed", value=42)
        else:
            self.key_test = test_col.selectbox("Select test set", file_options, index=None )
            
        
class SKMTab(SourceTab):
    
    name = "Scikit MultiLearn Dataset"
    source_label = "SKM"
    
    def render(self):
        avaiable_dataset = SKMultilearnDS.get_avaiable_datasets().keys()
        st.subheader("Select from Scikit MultiLearn Module")
        self.key_train = st.selectbox("SKM Datasets", avaiable_dataset, index=None )
        make_split = st.checkbox("Generate own train-test split", key="SKM_split", value=False)
        if make_split:
            self.seed = st.text_input("Radom Seed for dataset split", key="SKM_seed", value=42)

class HuggingFaceTab(SourceTab):
    
    name = "Hunggingface Datasets"
    source_label = "HF"
    
    def render(self):
        st.subheader("Load a dataset from Hugging Face Datasets Module") 
        key_col, info_col = st.columns(2)
        self.key_train = key_col.text_input("Dataset Name", placeholder="user/dataset")
        info_col.container(border=False, height=2)
        info_col.info("trust_remote_code is activated", icon="ℹ")
        
        split_options = ["train", "validation", "test", "train+test", "train+test+validation"]
        split_col, seed_col, _ = st.columns([3,2,2])
        self.df_split = split_col.selectbox("Dataset Split", split_options)
        
        make_split = split_col.checkbox("Generate own train-test split", key="HF_split", value=False)
        if make_split:
            self.seed = seed_col.text_input("Radom seed for split", key="HF_seed", value=42)
        
        st.text("Common Examples: qanastek/HoC")


def UploadBtn(method):
    if st.button("Upload", type="primary", key=method.name):
        datasets = st.session_state["session"].datasets
        params = method.get_params()
        datasets.load_dataset(method.source_label, params)
         
            
def column_selection(dataset):
    columns = dataset.get_columns()
    n_columns = len(columns)
    text_format = dataset.X_format
    label_format = dataset.y_format
    
    if len(columns):
        text_col, label_col =  st.columns(2)
        
        if text_format == "TEXT":
            X_col = text_col.selectbox("Select Text Column", range(n_columns), format_func=lambda i: columns[i], index=0 )
            X_info = (X_col, 1)
            
        elif text_format == "FREQ":
            col1, col2 = text_col.columns(2)
            X_features_start = col1.selectbox("Select Text Column", range(n_columns), format_func=lambda i: columns[i], index=0 )
            X_features_count = col2.number_input("N° Term Features", value=1)
            X_info = (X_features_start, X_features_count)
            
        if label_format in ["LIST", "LITERAL"]:  
            label_col = label_col.selectbox("Select Labels Column", range(n_columns), format_func=lambda i: columns[i], index=1)
            Y_info = (label_col, 1)
            
        if label_format == "BINARY":
            col1, col2 = label_col.columns(2)
            Y_features_start = col1.selectbox("Select Label Column", range(n_columns), format_func=lambda i: columns[i], index=1 )
            Y_features_count = col2.number_input("N° Labels", value=1)
            Y_info = (Y_features_start, Y_features_count)
            
        elif label_format == "SEP":
            col1, col2 = label_col.columns(2)
            Y_features_start = col1.selectbox("Select Label Column", range(n_columns), format_func=lambda i: columns[i], index=1 )
            dataset.format_sep = col2.text_input("Separator Character", value=',')
            Y_info = (Y_features_start, 1)
            
        
        dataset.columns_indices["X"] = X_info
        dataset.columns_indices["Y"] = Y_info
           
        drop_placeholder = []
        if "cols_to_drop" in dataset.__dict__.keys():
            drop_placeholder = dataset.cols_to_drop
        
        dataset.cols_to_drop = st.multiselect("Select Columns to Drop", columns, drop_placeholder, placeholder="None")
        
@st.dialog("Additional Dataset Settings", width="large")
def additional_settings(dataset):
    
    text_col, label_col, _ = st.columns([1, 1, 3])
    avaible_text_format = ["TEXT", "FREQ"]
    placeholder = avaible_text_format.index(dataset.X_format)
    dataset.X_format = text_col.selectbox("Text data format", avaible_text_format, index=placeholder )
    
    avaible_label_format = ["LIST", "BINARY", "SEP", "LITERAL"]
    placeholder = avaible_label_format.index(dataset.y_format)
    dataset.y_format = label_col.selectbox("Labels format", avaible_label_format, index=placeholder )
    
    st.divider()
    if not dataset.source == "SCIKIT_MULTILEARN":
        column_selection(dataset)
    
def loaded_datasets():
    datasets = st.session_state["session"].datasets
    with st.container(border=True):
        
        if len(datasets.items) == 0:
            st.warning("There is no datasets provided.")
            
        for i, dataset in enumerate(datasets.items):
            info, delete, config = st.columns([5,1,1])
            info.markdown(f"**[{i+1}] {dataset.name}**")
            info.markdown(f"Source: {dataset.source}")
            
            if delete.button(label="🗑", type="primary", key=f"{dataset.name}-DELBTN", use_container_width=True):
                datasets.remove(dataset)
                st.rerun()
                
            if config.button(label="⚙", type="primary", key=f"{dataset.name}-CONFIGBTN", use_container_width=True):
                additional_settings(dataset)
                
            if i < len(datasets.items) - 1:
                st.divider()

def RenderPage():
    load_methods =  {
        "Upload file":  FileTab,
        "Data Folder":  DataFolderTab,
        "Scikit MultiLearn":  SKMTab,
        "Hugging Face": HuggingFaceTab,
    } 
    
    with st.container(border=False, height=450):
        for method, tab in zip( load_methods.values(), st.tabs(load_methods.keys()) ):
            with tab:
                load_method = method()
                load_method.render()
                UploadBtn(load_method)
                
    loaded_datasets()

def DatasetPage():
    st.title("Load the Dataset")
    RenderPage()
    
if __name__ == "__main__": 
    menu()
    DatasetPage()