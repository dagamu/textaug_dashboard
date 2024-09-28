import pandas as pd
import importlib
import os

class FileTab:
    def __init__(self):
        self.name = "FileUpload"
        
    def get_data(self):
        filename = self.filepath.name
        return pd.read_csv(self.filepath), filename
    
    def render(self, st):
        st.subheader("Load a CSV file")
        self.filepath = st.file_uploader( "Choose a CSV file", accept_multiple_files=False )

class DataFolderTab:
    def __init__(self):
        self.name = "DataFolder"
        
    def get_data(self):
        return pd.read_csv(f"data/{self.filename}"), self.filename
    
    def render(self, st):
        st.subheader("Select from data folder")
        
        file_options = [ file for file in os.listdir("data") if file.endswith(".csv") ]
        self.filename = st.selectbox("CSV files in /data folder", file_options, index=None )

class HuggingFaceTab:
    def __init__(self):
        self.name = "Hugging Face"
        
    def get_data(self):
        load_dataset = importlib.import_module("datasets").load_dataset
        df_name = self.df_key.split('/')[1]
        df = load_dataset(self.df_key, split=self.df_split, trust_remote_code=True).to_pandas()
        
        return df, df_name
    
    def render(self, st):
        st.subheader("Load a dataset from Hugging Face Datasets Module") 
        
        st.info("trust_remote_code is activated", icon="ℹ")
        
        self.df_split = st.selectbox("Dataset Split", ["train", "validation", "test", "train+test", "train+test+validation"])
        self.df_key = st.text_input("Dataset Name", placeholder="user/dataset")
        st.text("Common Examples: qanastek/HoC")


def UploadBtn(st, method):
    btn_text = "Upload another dataset" if "df" in st.session_state else "Upload"
    if st.button(btn_text, type="primary", key=method.name):
        st.toast('Loading Dataset')
        df, df_name = method.get_data()
        st.toast('Done!')
        
        st.session_state['df_name'] = df_name
        st.session_state['df'] = df
         
        ## Clear prev properties
        for prop in ['labels_column', 'all_labels']:
            if prop in st.session_state:
                del st.session_state[prop]

def DatasetPage(st):
    
    st.title("Load the Dataset")

    if "df" in st.session_state:
        st.markdown(f"**Selected Dataset: {st.session_state['df_name']}**" )
    
    load_methods =  {
        "Upload file":  FileTab,
        "Data Folder":  DataFolderTab,
        "Hugging Face": HuggingFaceTab,
    } 
    
    for method, tab in zip( load_methods.values(), st.tabs(load_methods.keys()) ):
        with tab:
            load_method = method()
            load_method.render(st)
            UploadBtn( st, load_method )
    