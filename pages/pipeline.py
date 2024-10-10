import streamlit as st
from menu import menu

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
import pandas as pd

import random
import time

from pages.classification_model import VECTORIZER_METHODS, PROBLEM_TRANSFORM_METHODS, AVAIBLE_MODELS
from pages.classification_model import train_model, get_performance

from pipeline.dataset_list import DatasetList
from pipeline.aug_list import DataAugmentationList

from pages.data_augmentation import AugmentDataframe
    
def PipelineRun(datasets, vec_methods, pt_methods, models, aug_methods):
    
    results = []
    total_cases = len(datasets) * len(vec_methods) * len(pt_methods) * len(models) * len(aug_methods)
    
    prog_bar = st.progress(0, text="Progress Bar")
    
    i = 0
    for dataset in datasets:
        txt_col_name = dataset["text_column"]
        label_col_name = dataset["labels_column"]
        
        for vec_method in vec_methods:
            vectorizer = VECTORIZER_METHODS[vec_method]()
            
            for pt_method in pt_methods:
                multi_model, preprocessing = PROBLEM_TRANSFORM_METHODS[pt_method].values()
                preprocessing = preprocessing()
                
                for model in models:
                    
                    df = dataset["df"]
                    txt_col = dataset["text_proc_action"].apply_to(df[txt_col_name])
                    labels_col = dataset["labels_proc_action"].apply_to(df[label_col_name])
                    
                    
                    base_model, model_params = AVAIBLE_MODELS[ str(model) ]
                    base_model = base_model(**model_params)
                    X_train, X_test, y_train, y_test = train_test_split( txt_col, labels_col, test_size=0.3, random_state=42 )
                    
                    clf = train_model(vectorizer, multi_model, preprocessing, base_model, X_train, y_train )
                    
                    train_per = get_performance( clf, preprocessing, X_train, y_train )
                    test_per = get_performance( clf, preprocessing, X_test, y_test )
                    
                    df_to_aug = pd.DataFrame({"text_column": X_train, "labels_column": y_train} )
                            
                    for aug_method in aug_methods:
                        
                        time_performance_start = time.time()
                
                        aug_df, _ = AugmentDataframe( df_to_aug.copy(), "text_column", "labels_column", aug_method["method"] )
        
                        base_model, model_params = AVAIBLE_MODELS[ str(model) ]
                        base_model = base_model(**model_params)
                        
                        X_aug = aug_df["text_column"]
                        y_aug = aug_df["labels_column"]
                        
                        aug_clf = train_model(vectorizer, multi_model, preprocessing, base_model, X_aug, y_aug )
                        
                        aug_train_per = get_performance( aug_clf, preprocessing, X_aug, y_aug )
                        aug_test_per = get_performance( aug_clf, preprocessing, X_test, y_test )
                        
                        time_performance_end = time.time()
                        
                        results.append({
                            "df": dataset['name'],
                            "vec_method": vec_method,
                            "pt_method": pt_method,
                            "model": model,
                            "base_train_acc": round(train_per["acc"]*100, 2),
                            "base_test_acc": round(test_per["acc"]*100, 2),
                            "base_train_hl": round(train_per["hl"], 3),
                            "base_test_hl": round(test_per["hl"], 3),
                            "aug_method": aug_method['label'],
                            "aug_train_acc": round(aug_train_per["acc"]*100, 2),
                            "aug_test_acc": round(aug_test_per["acc"]*100, 2),
                            "aug_train_hl": round(aug_train_per["hl"], 3),
                            "aug_test_hl": round(aug_test_per["hl"], 3),
                            "time_performace": time_performance_end - time_performance_start
                        })
                        
                        i += 1
                        prog_text = f"{dataset['name']} - {vec_method} - {pt_method} - {model} - {aug_method['label']} - {i}/{total_cases} {100*i/total_cases:.2f}%"
                        prog_bar.progress(i/total_cases, text=prog_text)
                        #st.write(f"{} - {vec_method} - {pt_method} - {model} - {aug_method['label']}")    
    
    st.dataframe(results)
    
    
def PipelinePage():
        
    st.subheader("Pipeline Flow")
    
    if "pipeline_datasets" in st.session_state:
        selected_datasets = st.session_state["pipeline_datasets"]
    else:
        selected_datasets = DatasetList(st)
        
    selected_datasets.render()
    
    st.markdown("**Classification Models**")
    cf_container = st.container(border=True)
    
    vec_col, pt_col, model_col =  cf_container.columns(3)
    selected_vec = vec_col.multiselect("Vectorizer Methods", VECTORIZER_METHODS)
    selected_pt = pt_col.multiselect("Problem Transformation Methods", PROBLEM_TRANSFORM_METHODS)
    selected_models = model_col.multiselect("Classification Models", AVAIBLE_MODELS)
    
    if "pipeline_aug_methods" in st.session_state:
        selected_aug_methods = st.session_state["pipeline_aug_methods"]
    else:
        selected_aug_methods = DataAugmentationList()
    selected_aug_methods.render()
    
    if st.button("Run"):
        PipelineRun(selected_datasets.datasets, selected_vec, selected_pt, selected_models, selected_aug_methods.methods )

    
if __name__ == "__main__": 
    PipelinePage()
    menu()