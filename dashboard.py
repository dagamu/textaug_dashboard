import streamlit as st
from menu import menu

with open('README.md', 'r') as file:
    content = file.read()
    st.markdown(content) 
    
menu()
    