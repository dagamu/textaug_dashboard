import pandas as pd
import streamlit as st
from menu import menu
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from neattext.functions import clean_text
import ast

stop_words = set(stopwords.words('english'))

class ReplaceAction:

    name = "Value Replacing"
        
    def render(self, st, df):
        st.subheader("Value Replacing")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_column = st.selectbox("Select a Column", df.columns, index=None, key=self.name )
            if selected_column:
                if st.button("Replace"):
                    selected_column = None
                    selected_value = None
                    fun = lambda value: new_value if value == selected_value else value
                    df[selected_column] = df[selected_column].apply(fun)
        
        with col2:
            if selected_column:
                col = df[selected_column]
                selected_value = st.selectbox("Select a value to replace", pd.unique(col), index=None )
                
        with col3:
            if selected_column:      
                new_value = st.text_input("New Value")

class CustomFnAction:

    name = "Custom Text Processing"
    stemmer = PorterStemmer()
    
    def data_processing(self, text):
        text= text.lower()
        text = re.sub('<br />', '', text)
        text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    def stemming(self, data):
        text = [self.stemmer.stem(word) for word in data]
        return text
        
    def render(self, st, df):
        st.subheader("Text Pre-processing")
        selected_text_column = st.selectbox("Select a Column to apply function", df.columns, index=None, key="customfn" )
        
        if st.button("Apply"):
            df[selected_text_column] = df[selected_text_column].apply(lambda x: self.data_processing(x) )
            df = df.drop_duplicates(selected_text_column)
            df[selected_text_column] = df[selected_text_column].apply(self.stemming)

        
class NeatTextCleanAction:

    name = "NeatText Clean"
        
    def render(self, st, df):
        st.subheader("Neattext Clean Text Function")
        selected_text_column_nt = st.selectbox("Select a Column to apply function", df.columns, index=None, key="ntclean" )
        if st.button("Clean Text"):
            df[selected_text_column_nt] = df[selected_text_column_nt].apply( clean_text )
            st.toast("Done! See Dataview page to see changes")

class ListSplitAction:

    name = "List Split"
    
    def split_char(self, df, col, char=','):
        df[col] = df[col].str.split(char)
        
    def literal_split(self, df, col):
        df[col] = df[col].apply( ast.literal_eval )
        
    def render(self, st, df):
        st.info("Only list values in the format [ label1, label2, label3 ... ] will be accepted", icon="ℹ")
        selected_text_column = st.selectbox("Select a Column to apply function", df.columns, index=None, key="list_split" )
        
        split_methods = {
            "Literal List": self.literal_split,
            "Separate by coma (,)": self.split_char,
            "Separante by semicolon (;)": lambda col: self.split_char(col, char=';')
        }
        
        selected_method = st.selectbox("Select Method", split_methods.keys() )
        if st.button("List Split"):
            split_methods[selected_method]( df, selected_text_column )
            
TEXT_ACTIONS = [NeatTextCleanAction, ReplaceAction, CustomFnAction]
LABELS_ACTIONS = [ListSplitAction]

def TextProcessingPage():
    if not "df" in st.session_state:
        st.warning('There is no dataset :(') 
        return
    df = st.session_state["df"]
    
    st.title("Text Pre-processing")
    
    preprocessing_actions = [ action() for action in [ *LABELS_ACTIONS, *TEXT_ACTIONS ] ] 
    action_labels = [ action.name for action in preprocessing_actions ]
    for action, tab, in zip( preprocessing_actions, st.tabs(action_labels) ):
        with tab:
            action.render(st, df)
            
TextProcessingPage()
menu()