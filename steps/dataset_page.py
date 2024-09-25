import pandas as pd
import importlib
import os

def UploadTab(st):
    st.subheader("Load a CSV file")
    return st.file_uploader( "Choose a CSV file", accept_multiple_files=False ) 

def DataFolderTab(st):
    st.subheader("Select from data folder")
    file_options = [ file for file in os.listdir("data") if file.endswith(".csv") ]
    return st.selectbox("CSV files in /data folder", file_options, index=None ) 

def HuggingFaceTab(st):
    st.subheader("Load a dataset from Hugging Face Datasets Module") 
    st.info("trust_remote_code is activated", icon="ℹ")
    df_split = st.selectbox("Dataset Split", ["train", "validation", "test", "train+test", "train+test+validation"])
    df_name = st.text_input("Dataset Name", placeholder="user/dataset")
    st.text("Common Examples: qanastek/HoC")
    return df_split, df_name
    

def DatasetPage(st):
    
    st.title("Load the Dataset")

    if "df" in st.session_state:
        st.markdown(f"**Selected Dataset: {st.session_state['df_name']}**" )
    
    upload_tab, data_folder_tab, hf_tab = st.tabs([ "Upload file", "Data Folder", "Hugging Face" ])
    
    with upload_tab:
        uploaded_file = UploadTab(st)
        
    with data_folder_tab:
        selected_data = DataFolderTab(st)
    
    with hf_tab:
        hf_split, hf_dataset = HuggingFaceTab(st)
    
    
    st.text("")
    
    btn_text = "Upload another dataset" if "df" in st.session_state else "Upload"
    if st.button(btn_text, type="primary"):
        
        
        if hf_dataset:
            load_dataset = importlib.import_module("datasets").load_dataset
            df_name = hf_dataset.split('/')[1]
            st.toast('Loading Dataset')
            df = load_dataset(hf_dataset, split=hf_split, trust_remote_code=True).to_pandas()
            st.toast("Done!")
        
        else:
            if uploaded_file:
                csv_path = uploaded_file
                df_name = uploaded_file.name
                
            else:
                csv_path = f"data/{selected_data}"
                df_name = selected_data
                
            st.toast('Loading Dataset')
            df = pd.read_csv(csv_path)
            st.toast("Done!")
        
        st.session_state['df_name'] = df_name
        st.session_state['df'] = df
        
        
        ## Clear prev properties
        for prop in ['labels_column', 'all_labels']:
            if prop in st.session_state:
                del st.session_state[prop]