import streamlit as st
from menu import menu

from api.pipeline.dataset_list import DatasetList
from api.pipeline.aug_list import DataAugmentationList
from api.pipeline.aug_report import make_report
from api.pipeline.resampling_list import ResamplingList
from api.pipeline.runner import PipelineRunner
from pages.classification_model import VECTORIZER_METHODS, PROBLEM_TRANSFORM_METHODS, AVAIBLE_MODELS

def PipelinePage():
        
    st.subheader("Pipeline Flow")
    
    if "pipeline_datasets" in st.session_state:
        selected_datasets = st.session_state["pipeline_datasets"]
    else:
        selected_datasets = DatasetList(st)
        
    selected_datasets.render()
    
    if "pipeline_samplers" in st.session_state:
        selected_resampling = st.session_state["pipeline_samplers"]
    else:
        selected_resampling = ResamplingList()
    
    selected_resampling.render()
    
    st.markdown("**Classification Models**")
    cf_container = st.container(border=True)
    
    vec_col, pt_col, model_col =  cf_container.columns(3)
    selected_vec = vec_col.multiselect("Vectorizer Methods", VECTORIZER_METHODS )
    selected_pt = pt_col.multiselect("Problem Transformation Methods", PROBLEM_TRANSFORM_METHODS )
    selected_models = model_col.multiselect("Classification Models", AVAIBLE_MODELS )
    
    if "pipeline_aug_methods" in st.session_state:
        selected_aug_methods = st.session_state["pipeline_aug_methods"]
    else:
        selected_aug_methods = DataAugmentationList()
    selected_aug_methods.render()
    
    if st.button("Run"):
        runner = PipelineRunner(
            datasets=selected_datasets.datasets,
            resampling=selected_resampling.methods,
            vectorizers=selected_vec,
            problem_transformations=selected_pt,
            models=selected_models,
            aug_methods=selected_aug_methods
        )
        
        prog_bar = st.progress(0, text="Progress Bar")
        update = lambda p, text: prog_bar.progress( min(0.9999, p), text=text )
        runner.run(update)
        
        st.session_state["pipeline_results"] = runner.results

    if "pipeline_results" in st.session_state.keys():
        make_report(st.session_state["pipeline_results"])
    
if __name__ == "__main__": 
    PipelinePage()
    menu()