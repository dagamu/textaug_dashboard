import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MultiLabelBinarizer

import streamlit as st
from menu import menu

from src.imbalance_report import create_imbalance_report

def SummaryTab(dataset, report):
    
    labels = np.array(dataset.labels)
    max_count_labels = labels[ report["max_count"] ]
    min_count_labels = labels[ report["min_count"] ]
    max_count = report["labels_freq"][ report["max_count"] ]
    min_count = report["labels_freq"][ report["min_count"] ]
    
    table_sorted = pd.DataFrame( {"Label": dataset.labels, "Frequency": report["labels_freq"] } )
    table_sorted = table_sorted.sort_values(by="Frequency", ignore_index=True )
    
    fig = go.Figure()
    fig.add_trace( go.Scatter(x=table_sorted['Label'], y=table_sorted['Frequency'], mode='lines', name='Frequency'))
    fig.update_layout( xaxis_title="Label", yaxis_title="Frequency")
    st.plotly_chart( fig, config={'displayModeBar': False}, use_container_width=True )
    
    st.markdown(f"- Unique Labels: { len(dataset.labels) }\n")
    st.markdown(f"- More Occurrences: {max_count_labels} ({max_count})")
    st.markdown(f"- Less Occurrences: {min_count_labels} ({min_count})")
       
    
def BalaceTab(_, report):
    
    col1, col2, col3 = st.columns([1.5,1,1])
    with col1:
        st.markdown("</br>Mean Imbalance Ratio (MeanIR)", unsafe_allow_html=True)
        st.markdown("</br>Maximum Imbalance Ratio (MaxIR)", unsafe_allow_html=True)
        st.markdown("</br>Coefficient of variaton of IRLbl (CVIR)", unsafe_allow_html=True)
        
    with col2:
        st.latex(r'''\frac{1}{q}\sum_{\lambda\in L}IRLbl(\lambda)''')
        st.latex(r'''\max_{\lambda\in L}(IRLbl(\lambda))''')
        st.latex(r'''\frac {IRLbl\sigma}{MeanIR}''')
    
    with col3:
        st.markdown(f"</br>**{report['mean_ir']:.2f}**", unsafe_allow_html=True)
        st.markdown(f"</br>**{report['max_ir']:.2f}**", unsafe_allow_html=True)
        st.markdown(f"</br>**{report['cv_ir']:.2f}**", unsafe_allow_html=True)
        
def DataviewTab(dataset, report):
    labels_table = pd.DataFrame({ "Label": dataset.labels, "Frequency": report["labels_freq"], "Imbalance Ratio": report["irl_bl"]}) 
    st.dataframe(labels_table.round(decimals=2), width=500)

def DatasetReport(dataset, split):
    st.title("Label Analysis")
    
    if not dataset.loaded:
        dataset.get_data()
    
    if not dataset.preprocessed:
        session = st.session_state["session"]
        session.apply_preprocessing(dataset)
    
    sum_tab, balance_tab, dataview_tab = st.tabs(["Summary", "Balance Measures", "Dataview"])     
    y_features = dataset.y_train if split == "Train" else dataset.y_test
    
    mlb = MultiLabelBinarizer()
    y_features = mlb.fit_transform(y_features)
    imbalance_report = create_imbalance_report(y_features)
        
    with balance_tab:
        BalaceTab(dataset, imbalance_report)
    
    with sum_tab:
        SummaryTab(dataset, imbalance_report)    
        
    with dataview_tab:
        DataviewTab(dataset, imbalance_report)

def LabelAnalysisPage():
    
    datasets = st.session_state["session"].datasets
    if len(datasets.items):
        select_col, split_col, btn_col, _ = st.columns([2,2,1,1])
        selected_dataset = select_col.selectbox("Dataset to Analyse", options=datasets.items, format_func=lambda item: item.key_train)
        selected_split = split_col.selectbox("Split", options=["Train", "Test"], index=0)
        
        btn_col.container(height=10, border=False)
        if btn_col.button("View", type="primary"):
            DatasetReport(selected_dataset, selected_split)
            st.session_state["label_analysis_dataset"] = selected_dataset
            return
    else:
        st.warning("There are no datasets in the current session :(")
            
    if "label_analysis_dataset" in st.session_state:
        DatasetReport(selected_dataset, selected_split)
    
if __name__ == "__main__":          
    menu()
    LabelAnalysisPage()