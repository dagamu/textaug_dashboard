import streamlit as st
from cli_tools.parse_sesssion import read_default_session 

def menu():
    
    if not "session" in st.session_state:
        st.session_state["session"] = read_default_session()
    
    st.sidebar.header("Text Augmentation Dashboard")
    st.sidebar.divider()
    
    st.sidebar.page_link("pages/dataset_loading.py", label="🔼&nbsp; Dataset Loading" )
    st.sidebar.page_link("pages/dataview.py", label="🔎&nbsp; Dataview" )
    st.sidebar.page_link("pages/label_analysis.py", label="🏷️&nbsp; Label Analysis" )
    st.sidebar.page_link("pages/sampling.py", label="✂&nbsp; Sampling Methods" )
    #st.sidebar.page_link("pages/text_processing.py", label="🔡&nbsp; Text Processing" )
    st.sidebar.page_link("pages/classification_model.py", label="⚙️&nbsp; Classification Models" )
    st.sidebar.page_link("pages/data_augmentation.py", label="➕&nbsp; Data Augmentation" )
    st.sidebar.page_link("pages/pipeline.py", label="🧪&nbsp; Pipeline" )