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

class CustomFnAction:

    name = "Custom Text Processing"
    btn_label = "Apply"
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
        
    def apply_to(self, data):
        data = data.apply(lambda x: self.data_processing(x))
        data = data.apply(self.stemming)
        
    def render(self, st):
        st.subheader("Text Pre-processing")

        
class NeatTextCleanAction:

    name = "NeatText Clean"
    btn_label = "Clean"
    
    def apply_to(self, data ):
        return data.apply(clean_text)
        
    def render(self, st):
        st.subheader("Neattext Clean Text Function")

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
            
TEXT_ACTIONS = [NeatTextCleanAction, CustomFnAction]
LABELS_ACTIONS = [ListSplitAction]

def applyBtn(st, cols, action, set_action):
    _id = action.name
    if len(cols) > 1:
        selected_col = st.selectbox("Select a Column to apply function", cols, index=None, key=_id )
    else:
        selected_col = cols[0]
        
    if st.button(action.btn_label, key=_id+'btn' ):
        set_action(action, selected_col)
        
def RenderPage(st, cols, set_action):
    preprocessing_actions = [ action() for action in [ *LABELS_ACTIONS, *TEXT_ACTIONS ] ] 
    action_labels = [ action.name for action in preprocessing_actions ]
    for action, tab, in zip( preprocessing_actions, st.tabs(action_labels) ):
        with tab:
            action.render(st)
            applyBtn( st, cols, action, set_action)
    
def set_method_session(method, col):
    df = st.session_state["df"]
    df[col] = method.apply_to(df[col])

def TextProcessingPage():
    if not "df" in st.session_state:
        st.warning('There is no dataset :(') 
        return
    df = st.session_state["df"]
    
    st.title("Text Pre-processing")
    RenderPage(st, df.columns, set_method_session)
     
if __name__ == "__main__":     
    TextProcessingPage()
    menu()