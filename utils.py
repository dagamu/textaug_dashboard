import streamlit as st

LOADING_PRESETS = {
    "data_folder":
        {
            "Code Tutorials Multilabel.csv": {
                "cols_to_drop": ["mysql", "python", "php"]
            }
        }
}

def row_elements(st, elements, common_params={}):
    result = [None]* len(elements)
    for i, item, col in zip( range(len(elements)), elements, st.columns(len(elements)) ):
        with col:
            result[i] = item["el"](**item["params"], **common_params)
    return result

def custom_multiselect(label, default_options, initial_options, default_value, key):
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
    