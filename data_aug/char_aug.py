import streamlit as st
import nlpaug.augmenter.char as nac

class CharAugmenter:
    
    name = "Character Augmenter"
    
    CHAR_METHODS = { 
        "OCR":      ( nac.OcrAug, {} ),
        "Keyboard": ( nac.KeyboardAug, {} ),
        "Random":   ( nac.RandomCharAug, {} )
    }
    
    def renderOCR(self, stc):
        label = "Select a how many words would be modified"
        aug_word_min, aug_word_max = stc.slider(label, 0, 20, (1, 10))   
        return { "aug_word_min": aug_word_min, "aug_word_max": aug_word_max }
        
    def renderRandomMethod(self, stc):
        slider_label = "Select a how many words would be modified"
        aug_word_min, aug_word_max = stc.slider(slider_label, 0, 20, (1, 10))   
        action = stc.selectbox("Select Action", ["insert", "substitute", "swap", "delete"])
        
        upper_char = stc.checkbox("Include uppercase characters")
        num_char = stc.checkbox("Include numeric characters")

        return  { 
                    "action": action,
                    "aug_word_min": aug_word_min,
                    "aug_word_max": aug_word_max,
                    "include_numeric": upper_char,
                    "include_upper_case": num_char,
        }
        
        
    def renderKeyboardMethod(self, stc):
        slider_label = "Select a how many words would be modified"
        aug_word_min, aug_word_max = stc.slider(slider_label, 0, 20, (1, 10))   
            
        special_char = stc.checkbox("Include special characters")
        upper_char = stc.checkbox("Include uppercase characters")
        num_char = stc.checkbox("Include numeric characters")

        return { 
                    "aug_word_min": aug_word_min,
                    "aug_word_max": aug_word_max,
                    "include_special_char": special_char,
                    "include_numeric": upper_char,
                    "include_upper_case": num_char,
        }
    
    def render(self, stc):
        method = stc.selectbox("Method", self.CHAR_METHODS.keys() )
        
        renderMethods = { 
            "OCR": self.renderOCR,
            "Random": self.renderRandomMethod,
            "Keyboard": self.renderKeyboardMethod
        }
        additional_params = renderMethods[method](st)
        
        augmenter_class, initial_params = self.CHAR_METHODS[method]
        self.augmenter = augmenter_class( **initial_params, **additional_params ) 