import streamlit as st
from menu import menu

import ast

class ListSplitAction:

    name = "List Split"
    btn_label = "Split"
    
    def split_char(self, col, char=','):
        return col.str.split(char)
        
    def literal_split(self, col):
        return col.apply( ast.literal_eval )
        
    def apply_to(self, data):
        return self.split_methods[self.selected_method](data)
        
    def render(self, st):
        st.info("Only list values in the format [ label1, label2, label3 ... ] will be accepted", icon="ℹ")
        
        self.split_methods = {
            "Literal List": self.literal_split,
            "Separate by coma (,)": self.split_char,
            "Separante by semicolon (;)": lambda col: self.split_char(col, char=';')
        }
        
        self.selected_method = st.selectbox("Select Method", self.split_methods.keys() )
        
        
def RenderPage():
    preprocessing_manager = st.session_state["session"].preprocessing
    
def TextProcessingPage():
    
    st.title("Text Pre-processing")
    RenderPage()
     
if __name__ == "__main__":     
    menu()
    TextProcessingPage()