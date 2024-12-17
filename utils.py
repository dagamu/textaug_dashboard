import streamlit as st
from math import log10

LOADING_PRESETS = {
    "data_folder":
        {
            "Code Tutorials Multilabel.csv": {
                "cols_to_drop": ["mysql", "python", "php"]
            }
        }
}

def row_elements(elements, common_params={}):
    result = [None]* len(elements)
    for i, item, col in zip( range(len(elements)), elements, st.columns(len(elements)) ):
        with col:
            result[i] = item["el"](**item["params"], **common_params)
    return result

@st.fragment
def custom_multiselect(label, default_value, key):
    
    if not "custom_ms_cache" in st.session_state:
        st.session_state["custom_ms_cache"] = {}
    
    if not key in st.session_state["custom_ms_cache"].keys():
        st.session_state["custom_ms_cache"][key] = []
    
    current_items = st.session_state["custom_ms_cache"][key]
    def clear_items():
        st.session_state["custom_ms_cache"][key].clear()
    
    kind_col, multi_select_col, val_col, btn_col = st.columns([1,4,1,1])
    
    kind = kind_col.selectbox("Kind", ["%", "N°"], index=0, on_change=clear_items)
    
    st.session_state["custom_ms_cache"][key] = multi_select_col.multiselect(label, current_items, current_items )
        
    if kind == "N°":
        new_val = val_col.number_input( "New Value", value=default_value, min_value=1, key=f"{key}_input")
    else:
        default_value = default_value / 10 ** (int(log10(default_value)+1))
        new_val = val_col.number_input( "New Value", value=default_value, min_value=0.01, max_value=0.99, key=f"{key}_input")
        
    btn_col.container(border=False, height=11)
    if btn_col.button("Add", use_container_width=True, key=f"{key}Btn"):
        st.session_state["custom_ms_cache"][key].append(new_val)
        st.rerun()
        
    return st.session_state["custom_ms_cache"][key]

def custom_multiselect_old(label, default_options, initial_options, default_value, key):
    key = f"{key}_CustomMS"
    if not key in st.session_state.keys():
        st.session_state[key] = default_options
        
    multi_select_col, val_col, btn_col = st.columns([2,1,1])
    selected = multi_select_col.multiselect(label, st.session_state[key], initial_options )
    
    new_val = val_col.number_input( "New Value", value=default_value, key=f"{key}_input")
    btn_col.container(border=False, height=11)
    if btn_col.button("Add", use_container_width=True, key=f"{key}Btn"):
        st.session_state[key].append(new_val)
        st.rerun()
        
    return selected
    