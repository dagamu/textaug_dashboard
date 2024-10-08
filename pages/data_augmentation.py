import random

import pandas as pd
import streamlit as st
from menu import menu

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

from data_aug.eda_nlpaug import EDAug
from data_aug.char_aug import CharAugmenter

from utils import row_elements

#from transformers import GPT2LMHeadModel, GPT2Tokenizer

# https://nlpaug.readthedocs.io/en/latest/augmenter/augmenter.html  

class WordAugmenter:
    
    name = "Word Augmenter"
    
    def render(self, st):
        st.warning("Not avaible yet")
        return None
    
class SentenceAugmenter:
    
    name = "Sentence Augmenter"
    
    def generate_text(model, tokenizer, inital_text, n_words = 20 ):
        input_ids = tokenizer.encode(inital_text, return_tensors='pt')
        output = model.generate(input_ids, max_length=len(input_ids[0]) + n_words, num_return_sequences=1)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    def render(self, st):
        st.warning("Not avaible yet")
        return None
        llm_model = st.selectbox("Select Model", ["GPT2"])
        if llm_model == "GPT2":
            if st.button("Apply Augmentation", key="sentenceaug"):
                #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                #model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
                labels_colum = st.session_state["labels_column"]
                prompt = f"{df[labels_colum][0]} {df[text_column][0]} "
                st.text(prompt)
            #output = generate_text( model, tokenizer, )
        
class EDAaugmenter:
    
    name = "EDA Augmenter"
    
    eda_labels = [
        "Synonym Replacement (SR)",
        "Random Insertion (RI)",
        "Random Swap (RS)",
        "Random Deletion (RD)"
    ]
    
    def render(self, st):
        st.text("Weights of Easy Augmentation Methods")
        
        weights_items = [ { "params": {"label": label}, "el": st.number_input } for label in self.eda_labels ]
        common_params = {"value":1, "min_value":0 }
        eda_weights = row_elements( st, weights_items, common_params=common_params)
            
        self.augmenter = EDAug(p_weights=eda_weights)
        
def AugmentDataframe(df, text_col, labels_col, aug_method ):

    calc_samples = aug_method.calc_n_samples
    augmenter = aug_method.augmenter
    
    n_samples = calc_samples(len(df))
    
    df_dict = df[[text_col, labels_col]].to_dict("records")
    input_samples = random.choices( df_dict, k=n_samples )
    input_samples = pd.DataFrame( input_samples )
    input_samples = input_samples.rename( columns={ text_col: "input_sample" } )
    
    augmented_samples = augmenter.augment( list(input_samples["input_sample"].values) )
    input_samples["output_sample"] = augmented_samples
    
    new_samples = input_samples[["output_sample", labels_col]].rename(columns={ "output_sample": text_col })
    df = pd.concat([df, new_samples ], axis=0 )
    return df, input_samples
    
def SetupAugmenter(aug_method, set_method, unique_df):
    
    if "augmenter" not in vars(aug_method):
        return
    
    col1, col2, col3 = st.columns(3)
    
    _id = aug_method.name
    count_or_ratio = col1.selectbox("Samples to add", ["Count", "Ratio"], key=_id+'cr')
    
    if count_or_ratio == "Count":
        nsamples_count = col2.number_input("N° of samples to add", value=100, min_value=0, key=_id+'c')
        aug_method.calc_n_samples = lambda n: nsamples_count
    elif count_or_ratio == "Ratio":
        nsamples_ratio = col2.number_input("Ratio of samples to add", value=1.1, min_value=1.0, key=_id+'r')
        aug_method.calc_n_samples = lambda n: round( (nsamples_ratio-1)*n )

    if count_or_ratio == "Ratio" and unique_df and "df" in st.session_state:
        df = st.session_state["df"]
        nsamples_count = round( (nsamples_ratio - 1)*len(df) )
        
    if unique_df:
        aug_method_info = f"{aug_method.name} - {nsamples_count}"
        col3.caption(aug_method_info)
        btn_label = "Run"
    else:
        col3.caption(aug_method.name)
        btn_label = "Initialize"

    
    if col3.button(btn_label, key=_id+'i'):
        aug_method.params = { "n_input_samples": nsamples_count } if unique_df else {}
        set_method(aug_method)


def RenderPage(st, set_method, unique_df=True):
    
    methods = { 
        "Character Augmenter":      CharAugmenter,
        "Easy Data Augmentation":   EDAaugmenter,
        "Word Augmenter (*)":       WordAugmenter,
        "Sentecence Augmenter (*)": SentenceAugmenter
    }
    
    for tab, augmenter in zip( st.tabs(methods.keys()), methods.values() ):
        with tab:
            method_instance = augmenter()
            method_instance.render(st)
            SetupAugmenter(method_instance, set_method, unique_df)
                

def DataAugmentationPage():
    
    if not "df" in st.session_state:
        st.warning('There is no dataset :(') 
        return
    df = st.session_state["df"]
    
    st.title("Data Augmentation")
    labels_column = st.session_state["labels_column"]
    
    text_column = st.selectbox("Select Text Column", df.columns, index=0 )
    if text_column:
                
        augmenter = { "method": None }
        def set_aug_method(method):
            augmenter["method"] = method
            new_df, df_aug = AugmentDataframe(df, text_column, labels_column, method )
            st.table(df_aug)
            st.session_state["df"] = new_df
            st.toast(len(df))
            
        RenderPage(st, set_aug_method)
       
if __name__ == "__main__":          
    DataAugmentationPage()
    menu()