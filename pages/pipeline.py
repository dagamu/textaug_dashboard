import streamlit as st
from menu import menu

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
import pandas as pd

import random

from pages.text_processing import TEXT_ACTIONS, LABELS_ACTIONS
from pages.classification_model import VECTORIZER_METHODS, PROBLEM_TRANSFORM_METHODS, AVAIBLE_MODELS

from pipeline.dataset_list import DatasetList
from pipeline.aug_list import DataAugmentationList
    
def PipelineRun(datasets, vec_methods, pt_methods, models, aug_methods):
    
    results = []
    total_cases = len(datasets) * len(vec_methods) * len(pt_methods) * len(models) * len(aug_methods)
    
    prog_bar = st.progress(0, text="Progress Bar")
    
    i = 0
    for dataset in datasets:
        txt_col = dataset["text_column"]
        label_col = dataset["labels_column"]
        
        for vec_method in vec_methods:
            vectorizer = VECTORIZER_METHODS[vec_method]()
            
            for pt_method in pt_methods:
                multi_model, preprocessing = PROBLEM_TRANSFORM_METHODS[pt_method].values()
                preprocessing = preprocessing()
                
                for model in models:
                    working_df = dataset["df"].copy()
                    
                    X_features = vectorizer.fit_transform(working_df[txt_col])
                    y_features = preprocessing.fit_transform(working_df[label_col])
                    
                    X_train, X_test, y_train, y_test = train_test_split( X_features, y_features, test_size=0.3, random_state=42 )
                    
                    base_model = AVAIBLE_MODELS[model]()
                    clf = multi_model(base_model)
                    clf.fit(X_train, y_train)
                    
                    base_train_acc   = accuracy_score(   y_train, clf.predict(X_train) ) * 100
                    base_test_acc    = accuracy_score(   y_test , clf.predict(X_test ) ) * 100
                    base_train_hl    = hamming_loss(     y_train, clf.predict(X_train) )
                    base_test_hl     = hamming_loss(     y_test , clf.predict(X_test ) )
                            
                    for aug_method in aug_methods:
                        
                        augmenter = aug_method["method"].augmenter
                        
                        pool = working_df[[txt_col, label_col]].rename(columns={txt_col: "input_sample" })
                        n_samples = aug_method["method"].calc_n_samples(len(working_df))
                        input_samples = random.choices( pool.to_dict("records"), k=n_samples )
                        input_samples = pd.DataFrame(input_samples, columns=["input_sample", label_col])
                        augmented_samples = augmenter.augment( list(input_samples["input_sample"].values) )
                        input_samples["output_sample"] = augmented_samples
                        concat_df = input_samples[["output_sample", label_col]].rename(columns={ "output_sample": txt_col })
                        aug_df = pd.concat([working_df, concat_df ], axis=0 )
                        
                        X_aug_features = vectorizer.fit_transform(aug_df[txt_col])
                        y_aug_features = preprocessing.fit_transform(aug_df[label_col])
                        
                        base_model = AVAIBLE_MODELS[model]()
                        clf = multi_model(base_model)
                        clf.fit(X_aug_features, y_aug_features)
                        
                        X_features = vectorizer.transform(working_df[txt_col])
                        y_features = preprocessing.transform(working_df[label_col])
                        
                        X_train, X_test, y_train, y_test = train_test_split( X_features, y_features, test_size=0.3, random_state=42 )
                        
                        aug_train_acc   = accuracy_score(   y_train, clf.predict(X_train) ) * 100
                        aug_test_acc    = accuracy_score(   y_test , clf.predict(X_test ) ) * 100
                        aug_train_hl    = hamming_loss(     y_train, clf.predict(X_train) )
                        aug_test_hl     = hamming_loss(     y_test , clf.predict(X_test ) )
                        
                        results.append({
                            "df": dataset['name'],
                            "vec_method": vec_method,
                            "pt_method": pt_method,
                            "model": model,
                            "base_train_acc": round(base_train_acc, 2),
                            "base_test_acc": round(base_test_acc, 2),
                            "base_train_hl": round(base_train_hl, 3),
                            "base_test_hl": round(base_test_hl, 3),
                            "aug_method": aug_method['label'],
                            "aug_train_acc": round(aug_train_acc, 2),
                            "aug_test_acc": round(aug_test_acc, 2),
                            "aug_train_hl": round(aug_train_hl, 3),
                            "aug_test_hl": round(aug_test_hl, 3),
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