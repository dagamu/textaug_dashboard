import streamlit as st

def menu():
    
    st.sidebar.header("Text Augmentation Dashboard")
    
    st.sidebar.page_link("pages/dataset_loading.py", label="🔼 Dataset Loading" )
    st.sidebar.page_link("pages/dataview.py", label="🔎 Dataview" )
    st.sidebar.page_link("pages/text_processing.py", label="🔡 Text Processing" )
    st.sidebar.page_link("pages/label_analysis.py", label="🏷️ Label Analysis" )
    st.sidebar.page_link("pages/classification_model.py", label="⚙️ Classification Model" )
    st.sidebar.page_link("pages/data_augmentation.py", label="➕ Data Augmentation" )