import matplotlib.pyplot as plt
import streamlit as st
from menu import menu

# TODO: Mak filter views

def DataviewPage():
    if not "df" in st.session_state:
        st.warning('There is no dataset :(')
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
    DataviewPage()
    menu()