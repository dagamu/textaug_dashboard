import random

import pandas as pd

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

from eda_nlpaug.edaug import EDAug
from st_utils import row_elements

#from transformers import GPT2LMHeadModel, GPT2Tokenizer

# https://nlpaug.readthedocs.io/en/latest/augmenter/augmenter.html

CHAR_METHODS = { 
            "OCR":      ( nac.OcrAug, {} ),
            "Keyboard": ( nac.KeyboardAug, {} ),
            "Random":   ( nac.RandomCharAug, {} )
}

def CharAugTab(st):
    method = st.selectbox("Method", CHAR_METHODS.keys() )
    
    if method == "OCR":
        aug_word_min, aug_word_max = st.slider("Select a how many words would be modified", 0, 20, (1, 10))   
        additional_params = { "aug_word_min": aug_word_min, "aug_word_max": aug_word_max }
    
    elif method == "Random":
        aug_word_min, aug_word_max = st.slider("Select a how many words would be modified", 0, 20, (1, 10))   
        action = st.selectbox("Select Action", ["insert", "substitute", "swap", "delete"])
        
        upper_char = st.checkbox("Include uppercase characters")
        num_char = st.checkbox("Include numeric characters")

        additional_params = { 
                    "action": action,
                    "aug_word_min": aug_word_min,
                    "aug_word_max": aug_word_max,
                    "include_numeric": upper_char,
                    "include_upper_case": num_char,
        }
    
    elif method == "Keyboard":
        aug_word_min, aug_word_max = st.slider("Select a how many words would be modified", 0, 20, (1, 10))   
        
        special_char = st.checkbox("Include special characters")
        upper_char = st.checkbox("Include uppercase characters")
        num_char = st.checkbox("Include numeric characters")

        additional_params = { 
                    "aug_word_min": aug_word_min,
                    "aug_word_max": aug_word_max,
                    "include_special_char": special_char,
                    "include_numeric": upper_char,
                    "include_upper_case": num_char,
        }
        
    if method:
        augmenter_class, initial_params = CHAR_METHODS[method]
        augmenter = augmenter_class( **initial_params, **additional_params ) 
        return augmenter

    return None
    

def WordAugTab(st):
    st.warning("Not avaible yet")
    return None

def generate_text(model, tokenizer, inital_text, n_words = 20 ):
    input_ids = tokenizer.encode(inital_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=len(input_ids[0]) + n_words, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def SentenceAugTab(st):
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
        
def EDATab(st):
    st.text("Weights of Easy Augmentation Methods")
    eda_labels = [ "Synonym Replacement (SR)", "Random Insertion (RI)", "Random Swap (RS)", "Random Deletion (RD)" ]
    eda_weights = row_elements( st,
                 [ { "params": {"label": label}, "el": st.number_input } for label in eda_labels ],
                 common_params={"value":1, "min_value":0})
        
    augmenter = EDAug(p_weights=eda_weights)
    
    return augmenter

def DataAugmentationPage(st):
    
    if not "df" in st.session_state:
        st.warning('There is no dataset :(') 
        return
    df = st.session_state["df"]
    
    st.title("Data Augmentation")
    labels_column = st.session_state["labels_column"]
    
    text_column = st.selectbox("Select Text Column", df.columns, index=None )
    if text_column:
        
        tabs = { 
            "Character Augmenter":      CharAugTab,
            "Easy Data Augmentation":   EDATab,
            "Word Augmenter (*)":       WordAugTab,
            "Sentecence Augmenter (*)": SentenceAugTab
        }

        aug_method = None
        
        for tab, fn in zip( st.tabs(tabs.keys()), tabs.values() ):
            with tab:
                selected_aug_method = fn(st)
                if selected_aug_method != None:
                    aug_method = selected_aug_method
                    
        if aug_method:
            
            col1, col2, col3 = st.columns(3)
    
            aug_ready = False
    
            with col1:
                count_or_ratio = st.selectbox("Samples to add", ["Count", "Ratio"])
            
            with col2:
                if count_or_ratio == "Count":
                    new_samples_count = st.number_input("Number of samples to add", value=100, min_value=0)
                elif count_or_ratio == "Ratio":
                    new_samples_ratio = st.number_input("Ratio of samples to add", value=1.1, min_value=1.0)
                    
            with col3:
                n_input_samples = new_samples_count if count_or_ratio == "Count" else round( (new_samples_ratio - 1)*len(df) )
                aug_method_info = f"{aug_method.name} - {n_input_samples}"
                st.caption(aug_method_info)
                
                if st.button("Generate Augmentation"):
                    pool = df[[text_column, labels_column]].rename(columns={text_column: "input_sample" })
                    input_samples = random.choices( pool.to_dict("records"), k=n_input_samples )
                    input_samples = pd.DataFrame(input_samples, columns=["input_sample", labels_column])
                    augmented_samples = aug_method.augment( list(input_samples["input_sample"].values) )
                    input_samples["output_sample"] = augmented_samples
                    aug_ready = True
                    concat_df = input_samples[["output_sample", labels_column]].rename(columns={ "output_sample": text_column })
                    st.session_state["df"] = pd.concat([df, concat_df ], axis=0 )
            
            if aug_ready:        
                st.table(input_samples)
                
"""
st.subheader("Generate One Sample")

col1, col2 = st.columns(2)
text_param = ""
with col1:
    st.text_input("Original Text", value=text_param)
with col2:
    count_or_ratio = st.selectbox("EDA Method", ["SR", "RI", "RS", "RD"])
    
if st.button("Random Text"):
        text_param = random.choice( df[text_column] )
        
if st.button("Generate Sample"):
        pass
"""