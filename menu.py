import streamlit as st
from cli_tools.parse_sesssion import read_default_session 
from api.session import Session 

LOAD_DEFAULT = True

def menu():
    
    if not "session" in st.session_state:
        if LOAD_DEFAULT:
            st.session_state["session"] = read_default_session()
        else:
            st.session_state["session"] = Session()
    
    st.sidebar.header("Text Augmentation Dashboard")
    st.sidebar.divider()
    
    st.sidebar.page_link("pages/datasets_page.py", label="🔼&nbsp; Dataset Loading" )
    st.sidebar.page_link("pages/dataview_page.py", label="🔎&nbsp; Dataview" )
    st.sidebar.page_link("pages/imbalance_page.py", label="🏷️&nbsp; Label Analysis" )
    st.sidebar.page_link("pages/sampling_page.py", label="✂&nbsp; Sampling Methods" )
    #st.sidebar.page_link("pages/text_processing.py", label="🔡&nbsp; Text Processing" )
    st.sidebar.page_link("pages/classification_page.py", label="⚙️&nbsp; Classification Models" )
    st.sidebar.page_link("pages/augmentation_page.py", label="➕&nbsp; Data Augmentation" )
    st.sidebar.page_link("pages/pipeline_page.py", label="🧪&nbsp; Pipeline" )
    st.sidebar.page_link("pages/results_page.py", label="🗂&nbsp; Results" )