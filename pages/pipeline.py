import streamlit as st
from menu import menu

from pages.dataset_loading import loaded_datasets
from pages.sampling import render_sampling_methods
from pages.classification_model import selected_models
from pages.data_augmentation import RenderAugMethods, RenderAugSteps
from api.pipeline.aug_report import make_report

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
        session.pipeline_run(update)
    
        st.session_state["pipeline_results"] = session.runner.results

    if "pipeline_results" in st.session_state.keys():
        make_report(st.session_state["pipeline_results"])
        pass
    
if __name__ == "__main__": 
    menu()
    PipelinePage()