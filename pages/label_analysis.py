from math import trunc, log10, sqrt
from textwrap import dedent
import pandas as pd
import streamlit as st
from menu import menu

def extract_labels( df, labels_column ):
    all_labels = {}
    for i, row in df.iterrows():
        labels_text = row[labels_column]
        row_labels = labels_text
        
        if type(row_labels) == list:
            for l in row_labels:
                if l in all_labels:
                    all_labels[l]["count"] += 1
                else:
                    all_labels[l] = { "label": l, "count": 1 }
    return all_labels

def SummaryTab(st, all_labels):
        
    max_ = max( list(all_labels.keys()), key=lambda k: all_labels[k]["count"] )
    min_ = min( list(all_labels.keys()), key=lambda k: all_labels[k]["count"] )
    
    data = pd.DataFrame( [l.values() for l in all_labels.values()], columns=["Label", "Frequency", "Imbalance Ratio"] )
    data.sort_values(by="Frequency", inplace=True, ignore_index=True )
    
    max_log10 = trunc( log10(all_labels[max_]["count"]) )
    data['Order'] = data.index.astype(str)
    data['Index + Label'] = data["Order"].str.zfill(max_log10) + ' ' + data['Label'].astype(str)
    
    st.line_chart( data, x="Index + Label", y="Frequency"  )
    
    info = f"""\n\
        - Unique Labels: { len(all_labels.keys())}\n
        - More Occurrences: {max_} ({all_labels[max_]["count"]})\n
        - Less Occurrences: {min_} ({all_labels[min_]["count"]})"""
                
    st.markdown(dedent(info))
    
def BalaceTab(st, all_labels ):
    
    max_occurences = max( [label["count"] for label in all_labels.values()] )
    
    sum_IR = 0
    MaxIR = 0
    for label_data in all_labels.values():
        value = max_occurences / label_data["count"]
        all_labels[label_data["label"]]["imbalance_ratio"] = value
        sum_IR += value
        MaxIR = value if value > MaxIR else MaxIR
    
    q = len(all_labels)
    MeanIR = sum_IR / q
    
    IRLbl_sigma = sqrt( sum( [ (label["count"] - MeanIR)**2/(q-1) for label in all_labels.values()] ) )
    CVIR = IRLbl_sigma / MeanIR
    
    col1, col2, col3 = st.columns([1.5,1,1])
    with col1:
        st.markdown("</br>Mean Imbalance Ratio (MeanIR)", unsafe_allow_html=True)
        st.markdown("</br>Maximun Imbalance Ratio (MaxIR)", unsafe_allow_html=True)
        st.markdown("</br>Coefficient of variaton of IRLbl (CVIR)", unsafe_allow_html=True)
        
    with col2:
        st.latex(r'''\frac{1}{q}\sum_{\lambda\in L}IRLbl(\lambda)''')
        st.latex(r'''\max_{\lambda\in L}(IRLbl(\lambda))''')
        st.latex(r'''\frac {IRLbl\sigma}{MeanIR}''')
    
    with col3:
        st.markdown(f"</br>**{MeanIR:.2f}**", unsafe_allow_html=True)
        st.markdown(f"</br>**{MaxIR:.2f}**", unsafe_allow_html=True)
        st.markdown(f"</br>**{CVIR:.2f}**", unsafe_allow_html=True)
        
def DataviewTab(st, all_labels):
    labels_df = pd.DataFrame(all_labels.values())
    st.table(labels_df)


def LabelAnalysisPage():
    
    if not "df" in st.session_state:
        st.warning('There is no dataset :(')
        return
    df = st.session_state["df"]
    
    st.title("Label Analysis")
    
    labels_column = None
    if not "labels_column" in st.session_state and labels_column == None:
        selected_column = st.selectbox("Labels Column", df.columns, index=None )
        if selected_column:
            if st.button("Extract Labels", type="primary" ):
                labels_column = selected_column
                st.session_state["labels_column"] = selected_column
                st.session_state["labels_data"] = extract_labels(df, selected_column)
    else:
        labels_column = st.session_state["labels_column"]
                
    if not "all_labels" in st.session_state:
        st.session_state["labels_data"] = extract_labels(df, labels_column)
    
    if labels_column:    
        sum_tab, balance_tab, dataview_tab = st.tabs(["Summary", "Balance Measures", "Dataview"])     
        all_labels = st.session_state["labels_data"]
            
        with balance_tab:
            BalaceTab( st, all_labels )
        
        with sum_tab:
            SummaryTab( st, all_labels )    
            
        with dataview_tab:
            DataviewTab(st, all_labels)
   
if __name__ == "__main__":          
    LabelAnalysisPage()
    menu()