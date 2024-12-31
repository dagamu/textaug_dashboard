import streamlit as st
from menu import menu

from utils import row_elements

# https://nlpaug.readthedocs.io/en/latest/augmenter/augmenter.html  

class EDATab:
    
    key = "eda_aug"
    labels = [
        "Synonym Replacement (SR)",
        "Random Insertion (RI)",
        "Random Swap (RS)",
        "Random Deletion (RD)"
    ]
    
    def render(self):
        weights_items = [ { "params": { "label": label }, "el": st.number_input } for label in self.labels ]
        eda_weights = row_elements(weights_items, common_params={"value":1, "min_value":0})
        return { "p_weights": eda_weights }

class CharAugTab:
        
    key = "char_aug"
        
    def renderOCR(self):
        label = "Select a how many words would be modified"
        aug_word_min, aug_word_max = st.slider(label, 0, 20, (1, 10))   
        return { "aug_word_min": aug_word_min, "aug_word_max": aug_word_max }
        
    def renderRandomMethod(self):
        slider_label = "Select a how many words would be modified"
        aug_word_min, aug_word_max = st.slider(slider_label, 0, 20, (1, 10))   
        action = st.selectbox("Select Action", ["insert", "substitute", "swap", "delete"])
        
        upper_char = st.checkbox("Include uppercase characters")
        num_char = st.checkbox("Include numeric characters")

        return  { 
                    "action": action,
                    "aug_word_min": aug_word_min,
                    "aug_word_max": aug_word_max,
                    "include_numeric": upper_char,
                    "include_upper_case": num_char,
        }
        
        
    def renderKeyboardMethod(self):
        slider_label = "Select a how many words would be modified"
        aug_word_min, aug_word_max = st.slider(slider_label, 0, 20, (1, 10))   
            
        special_char = st.checkbox("Include special characters")
        upper_char = st.checkbox("Include uppercase characters")
        num_char = st.checkbox("Include numeric characters")

        return { 
                    "aug_word_min": aug_word_min,
                    "aug_word_max": aug_word_max,
                    "include_special_char": special_char,
                    "include_numeric": upper_char,
                    "include_upper_case": num_char,
        }
    
    def render(self):
            
        methods_dict = {
            "OCR": self.renderOCR,
            "keyboard": self.renderKeyboardMethod,
            "random": self.renderRandomMethod
        }
        method = st.selectbox("Method", methods_dict.keys(), format_func= lambda t: t.title()  )
        params = methods_dict[method]()
        return { "kind": method, "params": params } 
            
@st.fragment
def RenderAugSteps():
    st.markdown("**Augmentation Steps**")
    aug_manager = st.session_state["session"].aug_manager
    
    kind_col, steps_col, custom_val_col, custom_btn_col = st.columns([2,3,1,1])
         
    kind_labels = { "count": "Count (N)", "ratio": "Ratio (%)" }
    aug_manager.step_kind = kind_col.selectbox("Augmentation Kind", ["count", "ratio"], format_func=kind_labels.get)
    
    suffix = '%' if aug_manager.step_kind  == 'ratio' else ''
    format_steps = lambda val: f"+{val}{suffix}"
    aug_manager.steps = steps_col.multiselect("Steps", aug_manager.steps, aug_manager.steps, format_func=format_steps)
    
    val_args = { 
            "count": { "label": "Add Step (+N)", "value":   50,  "min_value": 1    , "step": 10, },
            "ratio": { "label": "Add Step (+%)", "value": 20.0, "min_value": 1.1, "step": 10.0 }
        }    
    custom_val = custom_val_col.number_input( **val_args[aug_manager.step_kind ] )
    
    custom_btn_col.container(height=12, border=False)
    if custom_btn_col.button("Add", use_container_width=True):
        if not custom_val in aug_manager.steps:
            aug_manager.steps.append(custom_val)
            st.rerun()
            
def RenderAugMethods():
    aug_manager = st.session_state["session"].aug_manager
    with st.container(border=True):
        
        if len(aug_manager.items) == 0:
            st.warning("There is no augmentation methods selected.")
            
        for i, item in enumerate(aug_manager.items):
            
            metadata, actions = st.columns([6,1])
            metadata.markdown(f"**[{i+1}] {item.name}**")
            
            if actions.button(label="🗑", type="primary", key=f"{item.name}{i}-DELBTN", use_container_width=True):
                aug_manager.remove(item)
                st.rerun()
            
            if i < len(aug_manager.items) - 1:
                st.divider()
            else:
                st.text("")

def DataAugmentationPage():
    st.title("Data Augmentation")
    
    aug_manager = st.session_state["session"].aug_manager
    method_tabs = { 
        "Character Augmenter":      CharAugTab,
        "Easy Data Augmentation":   EDATab,
    }
    
    for tab, augmenter in zip( st.tabs(method_tabs.keys()), method_tabs.values() ):
        with tab:
            method_instance = augmenter()
            params = method_instance.render()
            if st.button("Add Method", method_instance.key ):
                aug_manager.add_method( method_instance.key, params )
                
    st.divider()
    RenderAugSteps()
    
    st.divider()
    RenderAugMethods()

       
if __name__ == "__main__":          
    menu()
    DataAugmentationPage()