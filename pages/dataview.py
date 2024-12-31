import matplotlib.pyplot as plt
import streamlit as st
from menu import menu

import pandas as pd
import numpy as np

def trunc_features(X, max_):
    right = X[:, :max_//2]
    center = np.array( ['...'] * X.shape[0] )
    left = X[:, -max_//2:]
    return np.concatenate( (right, center[:, np.newaxis], left), axis=1)

def make_df(X_features, y_features):
    max_x_cols = 6
    max_y_cols = 4
    
    #X_view = trunc_features(X_features, max_x_cols) if X_features.shape[1] > max_x_cols else X_features
    #y_view = trunc_features(y_features, max_y_cols) if y_features.shape[1] > max_y_cols else y_features
    X_view = X_features
    y_view = y_features
    
    result_view = zip(X_view, y_view)
    return pd.DataFrame(result_view, columns=("Text", "Labels")) #list(range(result_view.shape[1]))

def view_dataset(dataset, split):
    if not dataset.loaded:
        dataset.get_data()
        st.text(f"{len(dataset.X_train)}, {len(dataset.y_train)}")
        
    if split == "Train":
        df = make_df(dataset.X_train, dataset.y_train)
    elif split == "Test":
        df = make_df(dataset.X_test, dataset.y_test)

        
    st.divider()
    query = st.text_input("Pandas Query [(*)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)")
    
    try:
        dataview = df.query(query)
    except:
        if len(query):
            st.toast("Invalid Query")
        dataview = df
        
    st.dataframe(dataview, use_container_width=True)

def DataviewPage():
    
    datasets = st.session_state["session"].datasets
    if len(datasets.items):
        select_col, split_col, btn_col, _ = st.columns([2,2,1,1])
        selected_dataset = select_col.selectbox("Dataset to View", options=datasets.items, format_func=lambda item: item.key_train)
        selected_split = split_col.selectbox("Split", options=["Train", "Test"], index=0)
        
        btn_col.container(height=10, border=False)
        if btn_col.button("View", type="primary"):
            view_dataset(selected_dataset, selected_split)
            st.session_state["dataview_dataset"] = selected_dataset
            return
    else:
        st.warning("There are no datasets in the current session :(")
            
    if "dataview_dataset" in st.session_state:
        view_dataset(selected_dataset, selected_split)
    
    return
    df = st.session_state["df"]
    text_col = st.session_state["text_col"]
    
    st.dataframe( df )
    st.divider()
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.text(f"N° Samples: {df.shape[0]}")
        
        if st.session_state["labels_data"]:
            st.text(f"Unique Labels: { len(st.session_state['labels_data'].keys())}")
        
        st.caption("Data Types")
        st.text( df.dtypes )
        
    with col2:
        fig, ax = plt.subplots()
        ax.set_title("Text Length", x=0.5, y=0.8)
        ax.hist( df[text_col].str.len() )
        fig.set_figheight(2.4)
        st.pyplot(fig, use_container_width=True)
 
if __name__ == "__main__":        
    menu()
    DataviewPage()