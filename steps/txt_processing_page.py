import pandas as pd
import re

import nltk
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from neattext.functions import clean_text
import ast

stop_words = set(stopwords.words('english'))

def data_processing(text):
    text= text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return text

def ReplaceTab(st, df):
    st.subheader("Value Replacing")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_column = st.selectbox("Select a Column", df.columns, index=None )
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

def CustomFnTab(st, df):
    st.subheader("Text Pre-processing")
    selected_text_column = st.selectbox("Select a Column to apply function", df.columns, index=None, key="customfn" )
    
    if st.button("Apply"):
        df[selected_text_column] = df[selected_text_column].apply(lambda x: data_processing(x) )
        df = df.drop_duplicates(selected_text_column)
        df[selected_text_column] = df[selected_text_column].apply(stemming)
        
        st.session_state["df"] = df
        
def NeatTextCleanTab(st, df):
    st.subheader("Neattext Clean Text Function")
    selected_text_column_nt = st.selectbox("Select a Column to apply function", df.columns, index=None, key="ntclean" )
    if st.button("Clean Text"):
        df[selected_text_column_nt] = df[selected_text_column_nt].apply( clean_text )
        st.session_state["df"] = df
        
        st.toast("Done! See Dataview page to see changes")

def ListSplitTab(st, df):
    st.subheader("List Split")
    st.info("Only list values in the format [ label1, label2, label3 ... ] will be accepted", icon="ℹ")
    selected_text_column = st.selectbox("Select a Column to apply function", df.columns, index=None, key="list_split" )
    split_method = st.selectbox("Select Method", ["Seperate by coma (,)", "Separate by semicolon (;)", "Literal List"] )
    if st.button("List Split"):
        if split_method == "Seperate by coma (,)":
            df[selected_text_column] = df[selected_text_column].str.split(",")
        elif split_method == "Separate by semicolon (;)":
            df[selected_text_column] = df[selected_text_column].str.split(";")
        elif split_method == "Literal List":
            df[selected_text_column] = df[selected_text_column].apply( ast.literal_eval )

def TextProcessingPage(st):
    if not "df" in st.session_state:
        st.warning('There is no dataset :(') 
        return
    df = st.session_state["df"]
    
    st.title("Text Pre-processing")
    
    ntclean_tab, list_split_tab, replace_tab, custom_fn_tab = st.tabs(
        ["NeatText Clean", "List Split", "Value Replacing", "Custom Text Processing"]
    )
    
    with ntclean_tab:
        NeatTextCleanTab(st, df)
    
    with list_split_tab:
        ListSplitTab(st, df)
    
    with replace_tab:
        ReplaceTab(st, df)
        
    with custom_fn_tab:
        CustomFnTab(st, df)