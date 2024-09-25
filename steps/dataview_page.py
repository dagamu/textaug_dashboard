def DataviewPage(st):
    if not "df" in st.session_state:
        st.warning('There is no dataset :(')
        return
    df = st.session_state["df"]
    
    st.dataframe( df )
    st.divider()
    st.text( f"Shape: {df.shape}")
    st.text( df.dtypes )
