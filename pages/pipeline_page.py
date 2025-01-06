import streamlit as st

from menu import menu
from pages.datasets_page import loaded_datasets
from pages.sampling_page import render_sampling_methods
from pages.classification_page import selected_models
from pages.augmentation_page import RenderAugMethods, RenderAugSteps

def PipelinePage():  
    st.subheader("Pipeline")
    
    session = st.session_state["session"]
    
    st.markdown("**Datasets**")
    loaded_datasets()
    
    st.markdown("**Sampling**")
    render_sampling_methods()
    
    st.markdown("**Classification Models**")
    selected_models()
    
    st.markdown("**Data Augmentation Methods**")
    RenderAugMethods()
    RenderAugSteps()
    
    if st.button("Run"):
        prog_bar = st.progress(0, text="Progress Bar")
        update = lambda p, text: prog_bar.progress( min(0.9999, p), text=text )
        session.report.set_results( session.pipeline_run(update) )
        st.page_link("pages/results_page.py", label="➡ Go to Results Page" )
    
if __name__ == "__main__": 
    menu()
    PipelinePage()