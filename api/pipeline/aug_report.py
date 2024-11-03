import streamlit as st
import pandas as pd
import math
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

COL_FORMAT_DICT = {
    "df": "Dataset",
    "vec_method": "Representation Method",
    "pt_method": "Problem Trasformation Method",
    "model": "Model",
    "aug_method": "Augmentation Method",
    "aug_choice_method": "Augmentation Input Choice Method" 
}

METRICS_FORMAT_DICT = {
    "acc": "Accuracy",
    "exact_acc": "Exact Accuracy",
}

def make_report(results):
    
    st.markdown("**Pipeline Results**")
    report_tab, table_tab = st.tabs(["Report", "Table"])
    
    results_df = pd.DataFrame(results)
    table_tab.dataframe(results_df)
    
    with report_tab:
        
        graph_col, setting_col = st.columns([5,2])
        
        with setting_col:
            
            num_cols = results_df._get_numeric_data().columns
            categorical_cols = set(results_df.columns) - set(num_cols)
            
            group_by_options = [None] + [ col for col in categorical_cols if len(results_df[col].unique()) > 1 ]
            group_by = st.selectbox("Group by", group_by_options, format_func=COL_FORMAT_DICT.get )
            
            lines = st.selectbox("Lines", [None] + list(categorical_cols) , format_func=COL_FORMAT_DICT.get)
            
            performance_metric = st.selectbox("Performance Metric", ["acc", "exact_acc"], format_func=METRICS_FORMAT_DICT.get)
            
        with graph_col:
            
            pattern = rf"^[^_]+_{performance_metric}$"
            metric_cols = [col for col in results_df.columns if re.search(pattern, col) ]
            format_cols = [ f"*{col.split('_')[0]}" for col in metric_cols ]
            
            min_val = 100
            max_val = 0
            
            if group_by == None:
                fig = go.Figure()
                y_values = results_df[metric_cols].mean()
                graph = go.Scatter( x=format_cols, y=y_values, mode='lines+markers' )
                fig.add_trace( graph )
            
            else:
                
                n_rows = math.ceil( len( results_df[group_by].unique() ) / 2 )
                fig = make_subplots( rows=n_rows, cols=2, subplot_titles=results_df[group_by].unique(), shared_yaxes=True )
                
                for i, group_value in enumerate( results_df[group_by].unique() ):
                    df = results_df[ results_df[group_by] == group_value ]
                    
                    if lines == None:
                        y_values = df[metric_cols].mean()
                        graph = go.Scatter( x=format_cols, y=y_values, mode="lines+markers"  ) 
                        fig.add_trace(graph, row=i//2+1, col=i%2+1)
                    else:
                        lines_unique = df[lines].unique()
                        colors = px.colors.qualitative.Plotly
                        for j, line_value in enumerate(lines_unique):
                            y_values = df[ df[lines] == line_value][metric_cols].mean()
                            graph = go.Scatter( x=format_cols, y=y_values,
                                               mode="lines+markers",
                                               name=line_value,
                                               line={"color": colors[j] })
                                               
                            fig.add_trace(graph, row=i//2+1, col=i%2+1)
                            
                fig.update_traces( showlegend=False )
                fig.update_traces( row=1, col=1, showlegend=lines!=None )
                    
            fig.update_layout( yaxis_title=METRICS_FORMAT_DICT[performance_metric] )
                
            st.plotly_chart( fig, config={'displayModeBar': True}, use_container_width=True )
                