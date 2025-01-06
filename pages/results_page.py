import streamlit as st
import math

from menu import menu

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class StreamlitReport:
    def debug(*args, **kwargs):
        st.json(*args, **kwargs)
    
    def title(text):
        st.markdown(f"**{text}**")
        
    def sections(section_list):
        return st.tabs(section_list)
    
    def columns(*args):
        return st.columns(*args)
    
    def selectbox(*args, **kargs):
        return st.selectbox(*args, **kargs)
    
    def line_chart(x, y):
        fig = go.Figure()
        graph = go.Scatter( x=x, y=y, mode='lines+markers' )
        fig.add_trace(graph)
        return fig
        
    def subplots(results_df, group_by, lines, metric_cols, format_cols):
        n_rows = math.ceil( len( results_df[group_by].unique() ) / 2 )
        fig = make_subplots( rows=n_rows, cols=2, subplot_titles=results_df[group_by].unique(), shared_yaxes=True )
        
        fig.add_annotation(text=f"Graph Title Test",
                  xref="paper", yref="paper",
                  x=0.5, y=1.15, showarrow=False,
                  font=dict(size=16))
        
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
        return fig
        
    def set_plots(fig, **kwargs):
        fig.update_layout(**kwargs) 
        st.plotly_chart( fig, config={'displayModeBar': True}, use_container_width=True )


def PipelinePage():
    
    st.subheader("Session Results")
    session = st.session_state["session"]
    
    if session.report.loaded:
        session.report.render(StreamlitReport)
    else:
        st.warning("No experiments have been carried out yet :(")
        st.page_link("pages/pipeline_page.py", label="➡ Go to Pipeline Page" )
    
if __name__ == "__main__": 
    menu()
    PipelinePage()