import streamlit as st
from menu import menu

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import time

from pages.classification_model import VECTORIZER_METHODS, PROBLEM_TRANSFORM_METHODS, AVAIBLE_MODELS
from pages.classification_model import train_model, get_performance

from src.genetic_sampler import GeneticSamplerSetup 
from api.pipeline.dataset_list import DatasetList
from api.pipeline.aug_list import DataAugmentationList
from api.pipeline.aug_report import make_report

from pages.data_augmentation import AugmentDataframe

from sklearn.preprocessing import MultiLabelBinarizer
    
def PipelineRun(datasets, genetic_sampler, vec_methods, pt_methods, models, aug_methods_list):
    
    aug_methods = aug_methods_list.methods
    aug_steps = np.sort( aug_methods_list.steps )
    aug_kind = aug_methods_list.aug_kind
    aug_selection = aug_methods_list.select_input_method
    
    results = []
    total_cases = len(datasets) * len(vec_methods) * len(pt_methods) * len(models) * len(aug_methods) 
    #prog_bar = st.progress(0, text="Progress Bar")
    
    mlb = MultiLabelBinarizer()
    
    i = 0
    for dataset in datasets:
        txt_col_name = dataset["text_column"]
        label_col_name = dataset["labels_column"]
        
        df = dataset["df"]
        txt_col = dataset["text_proc_action"].apply_to( df[txt_col_name] )
        labels_col = dataset["labels_proc_action"].apply_to( df[label_col_name] )
    
        y_features = mlb.fit_transform(labels_col)
        dataset_samples = genetic_sampler.generate_masks(y_features)
        
        for df_mask in dataset_samples:
            
            irlbl = genetic_sampler.calc_irlbl( df_mask )
            balance_metrics = {
                "mean_ir": round( np.mean(irlbl), 2),
                "max_ir": round( np.max(irlbl), 2)
            }
            
            X_train, X_test, y_train, y_test = train_test_split( txt_col[df_mask], labels_col[df_mask], test_size=0.3, random_state=42 )
            
            for vec_method in vec_methods:
                vectorizer = VECTORIZER_METHODS[vec_method]()
                
                for pt_method in pt_methods:
                    multi_model, preprocessing = PROBLEM_TRANSFORM_METHODS[pt_method].values()
                    preprocessing = preprocessing()
                    
                    for model in models:
                        base_model, model_params = AVAIBLE_MODELS[ str(model) ]
                        base_model = base_model(**model_params)
                        
                        preprocessing.fit(labels_col)
                        
                        if aug_kind == "count":
                            delta_steps = [ 0, aug_steps[0] ]
                            for j in range(1, len(aug_steps)):
                                value = aug_steps[j]
                                delta_steps.append( value - aug_steps[j-1] )
                                
                        elif aug_kind == "ratio":
                            to_ratio = lambda x: int(x/100 * X_train.shape[0] ) 
                            delta_steps = [ 0, to_ratio(aug_steps[0])]
                            for j in range(1, len(aug_steps)):
                                value = aug_steps[j]
                                delta_steps.append( int( to_ratio(value) - sum(delta_steps) )  )
                        
                        for aug_method in aug_methods:
                            
                            
                            for aug_selection_method in aug_selection:
                            
                                df_to_aug = pd.DataFrame({"text_column": X_train, "labels_column": y_train} )
                                result_row = {
                                    "df": dataset['name'],
                                    "n_samples": np.sum(df_mask),
                                    **balance_metrics,
                                    "vec_method": vec_method,
                                    "pt_method": pt_method,
                                    "model": model,
                                    "aug_method": aug_method['label'],
                                    "aug_choice_method": aug_selection_method
                                }
                                
                                for k, step in enumerate(delta_steps):
                                    time_performance_start = time.time()
                                    
                                    aug_method["method"].calc_n_samples = lambda _: step
                                    
                                    df_to_aug, _ = AugmentDataframe( df_to_aug, "text_column", "labels_column", aug_method["method"], aug_selection_method )
                                    
                                    base_model, model_params = AVAIBLE_MODELS[ str(model) ]
                                    base_model = base_model(**model_params)
                                    
                                    X_aug = df_to_aug["text_column"]
                                    y_aug = df_to_aug["labels_column"]

                                    aug_clf = train_model(vectorizer, multi_model, preprocessing, base_model, X_aug, y_aug )
                                    aug_performance = get_performance( aug_clf, preprocessing, X_test, y_test )
                                    
                                    suffix = '%' if aug_kind == 'ratio' else ''
                                    step_label = aug_steps[k-1] if k > 0 else "base"
                                    prefix = f"{'+' if k > 0 else ''}{step_label}{suffix}"
                                    
                                    result_row = {
                                        **result_row,
                                        f"{prefix}_train_samples": X_aug.shape[0],
                                        f"{prefix}_acc":        round( aug_performance["acc"] * 100,2) ,
                                        f"{prefix}_exact_acc":  round( aug_performance["exact_acc"] * 100,2),
                                        f"{prefix}_hl":         round( aug_performance["hl"] * 100,2),
                                        f"{prefix}_time_performace": time.time() - time_performance_start
                                    }
                                
                                results.append(result_row)
                                i += 1
                                prog_text = f"{dataset['name']} - {vec_method} - {pt_method} - {model} - {aug_method['label']} - {i}/{total_cases} {100*i/total_cases:.2f}%"
                                #prog_bar.progress(i/total_cases, text=prog_text)
    
    st.session_state["pipeline_results"] = results
    
    
    
def PipelinePage():
        
    st.subheader("Pipeline Flow")
    
    if "pipeline_datasets" in st.session_state:
        selected_datasets = st.session_state["pipeline_datasets"]
    else:
        selected_datasets = DatasetList(st)
        
    selected_datasets.render()
    
    genetic_sampler = GeneticSamplerSetup()
    genetic_sampler.render()
    
    st.markdown("**Classification Models**")
    cf_container = st.container(border=True)
    
    vec_col, pt_col, model_col =  cf_container.columns(3)
    selected_vec = vec_col.multiselect("Vectorizer Methods", VECTORIZER_METHODS )
    selected_pt = pt_col.multiselect("Problem Transformation Methods", PROBLEM_TRANSFORM_METHODS )
    selected_models = model_col.multiselect("Classification Models", AVAIBLE_MODELS )
    
    if "pipeline_aug_methods" in st.session_state:
        selected_aug_methods = st.session_state["pipeline_aug_methods"]
    else:
        selected_aug_methods = DataAugmentationList()
    selected_aug_methods.render()
    
    if st.button("Run"):
        PipelineRun(selected_datasets.datasets, genetic_sampler, selected_vec, selected_pt, selected_models, selected_aug_methods )

    if "pipeline_results" in st.session_state.keys():
        make_report(st.session_state["pipeline_results"])
    
if __name__ == "__main__": 
    PipelinePage()
    menu()