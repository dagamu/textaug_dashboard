import streamlit as st
from menu import menu

from utils import custom_multiselect

def RandomSampleTab():
    return {"sampling_list": custom_multiselect("Select N° Samples", 100, key="RandomSampler" )}
    
def FullDatasetTab():
    st.info("All samples from training set will be take for training.", icon="ℹ")
    return {}

def render_sampling_methods():
    sampling_manager = st.session_state["session"].sampling
    with st.container(border=True):
        for i, item in enumerate(sampling_manager.items):
            st.markdown(f"**[{i+1}] {item.name}**")
            if i < len(sampling_manager.items) - 1:
                st.divider()
            else:
                st.text("")

def SamplingPage():
    st.subheader("Sampling Methods")
    
    sampling_manager = st.session_state["session"].sampling
    sampling_tabs = {
        "Full Dataset": FullDatasetTab,
        "Random Sample": RandomSampleTab
    }
    sampling_names = sampling_tabs.keys()
    
    for key, tab in zip( sampling_names, st.tabs(sampling_names) ):
        with tab:
            if key in sampling_manager.available_methods:
                method_match = sampling_manager.available_methods[key]
                params = sampling_tabs[key]()
                if st.button("Add New Method", key=f"{key}_btn"):
                    sampling_manager.add_method(method_match(**params))
                    
    st.divider()
    render_sampling_methods()

if __name__ == "__main__":         
    menu()
    SamplingPage()