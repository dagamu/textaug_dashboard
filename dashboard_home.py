# Streamlit library required (!pip install streamlit)
# run as "streamlit run [file]" in command line or "python -m streamlit run [file]"

import streamlit as st
from streamlit_option_menu import option_menu

from steps.dataset_page import DatasetPage
from steps.dataview_page import DataviewPage
from steps.txt_processing_page import TextProcessingPage
from steps.label_analysis_page import LabelAnalysisPage
from steps.classifcation_model import ClasificationModelPage
from steps.data_augmentation_page import DataAugmentationPage

steps_pages = {
        "Load the Dataset": DatasetPage,
        "Dataview": DataviewPage,
        "Text Pre-processing": TextProcessingPage,
        "Label Analysis": LabelAnalysisPage,
        "Classification Model": ClasificationModelPage,
        "Data Augmentation": DataAugmentationPage,
    } 

with st.sidebar:
    selected = option_menu( menu_title = None, options = list(steps_pages.keys()) )  
   
for title, page in steps_pages.items():
    if selected == title:
        page(st)