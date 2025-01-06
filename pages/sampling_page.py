import streamlit as st
import numpy as np
from menu import menu

from utils import custom_multiselect

def RandomSampleTab():
    return {"sampling_list": custom_multiselect("Select N° Samples", 100, key="RandomSampler" )}
    
def FullDatasetTab():
    st.info("All samples from training set will be take for training.", icon="ℹ")
    return {}

def GeneticSamplerTab():
    
    def range_cases(label, key):
      with st.container():
              col1, col2 = st.columns([3,1])
              
              min_value, max_value = col1.slider(f"{label} ({key})", min_value=0, max_value=100, value=(0,100))
              n_cases = col2.number_input("N° Cases", min_value=1, key=key)
              
              range_ = [ "Free" ]
              if n_cases > 1 and min_value != max_value:
                step = (max_value-min_value)/(n_cases-1)
                range_ += np.arange( min_value, max_value+1, step=step ).tolist()
              else:
                mean_value = (max_value+min_value)/2
                range_ +=  [mean_value]
                
              formated_list =  ["Free"] + [ f"{ round(i, 2)}%" for i in range_[1:] ]
              col1.text( 'Cases: ' + ', '.join(formated_list) )
              return range_
    
    col1, col2, col3 = st.columns(3) 
    col3.container(border=False, height=10)
    return {
        "keep_labels": col3.checkbox("Keep N° of Labels"),
        "pob_size": col1.number_input("Population Size", min_value=5, value=20 ),
        "max_iterations": col2.number_input("Max Iterations", min_value=5, value=10 ),
        "n_samples_list": custom_multiselect("% of Samples", 100, "genetic_nsamples"),
        "MeanIrRange": range_cases("Mean Imbalance Ratio ", "MeanIR"),
        "MaxIrRange": range_cases("Max Imbalance Ratio", "MaxIR")
    }
    
def render_sampling_methods():
    sampling_manager = st.session_state["session"].sampling
    with st.container(border=True):
        for i, item in enumerate(sampling_manager.items):
            
            metadata, actions = st.columns([6,1])
            metadata.markdown(f"**[{i+1}] {item.name}**")
            
            if actions.button(label="🗑", type="primary", key=f"{item.name}-DELBTN", use_container_width=True):
                sampling_manager.remove(item)
                st.rerun()
            
            if i < len(sampling_manager.items) - 1:
                st.divider()
            else:
                st.text("")
                
        if len(sampling_manager.items) == 0:
            st.warning("There is no sampling methods provided.")

def SamplingPage():
    st.subheader("Sampling Methods")
    
    sampling_manager = st.session_state["session"].sampling
    sampling_tabs = {
        "Full Dataset": FullDatasetTab,
        "Random Sample": RandomSampleTab,
        "Genetic Algorithm": GeneticSamplerTab,
    }
    sampling_names = sampling_tabs.keys()
    
    for key, tab in zip( sampling_names, st.tabs(sampling_names) ):
        with tab:
            if key in sampling_manager.available_methods:
                method_match = sampling_manager.available_methods[key]
                params = sampling_tabs[key]()
                if st.button("Add New Method", key=f"{key}_btn"):
                    sampling_manager.add_method(method_match, params)
                    
    st.divider()
    render_sampling_methods()

if __name__ == "__main__":         
    menu()
    SamplingPage()