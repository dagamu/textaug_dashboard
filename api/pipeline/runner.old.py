import numpy as np
import pandas as pd

import time

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from pages.classification_model import VECTORIZER_METHODS, PROBLEM_TRANSFORM_METHODS, AVAIBLE_MODELS
from pages.data_augmentation import AugmentDataframe
from pages.classification_model import train_model
from src.lwise_performance import get_performance

class PipelineRunner:
    
    def __init__(self, datasets, resampling, vectorizers, problem_transformations, models, aug_methods ) -> None:
        self.datasets = datasets
        self.resampling_methods = resampling
        self.vectorizers = vectorizers
        self.problem_transformations = problem_transformations
        self.models = models
        self.aug_methods = aug_methods
        
        self.mlb = MultiLabelBinarizer()
        
    def run(self, update_fn):
        
        self.update_fn = update_fn
        self.results = []
        self.row_info = {}
        self.current_iteration = {}
        
        self.iter_count = 0
        self.total_iterations = self.calc_iterations()
        
        for dataset in self.datasets:
            for sampler in self.resampling_methods:
                
                df = dataset["df"]
                self.txt_col_name = dataset["text_column"]
                self.label_col_name = dataset["labels_column"]
                
                self.current_iteration["df"] = df
                self.row_info["df"] = dataset['name']
                
                labels_col = df[ dataset["labels_column"] ]
                if dataset["labels_proc_action"]:
                    labels_col = dataset["labels_proc_action"].apply_to(labels_col)
                    
                txt_col = df[ dataset["text_column"] ] 
                if dataset["text_proc_action"]:
                    txt_col = dataset["text_proc_action"].apply_to(txt_col)
                    
                y_features = self.mlb.fit_transform(labels_col)
                dataset_samples = sampler.get_sample(y_features)
                self.run_subset(dataset_samples)
                
    def run_subset(self, dataset_samples):
        df = self.current_iteration["df"]
        for df_mask in dataset_samples:
            df_mask = df_mask.astype(bool)            
            self.row_info["n_samples"] = np.sum(df_mask)
            
            text_col = df[self.txt_col_name][df_mask]
            labels_col = df[self.label_col_name][df_mask]
            
            split = train_test_split( text_col, labels_col, test_size=0.3, random_state=42 )
            self.run_training(df, split)
            
    def run_training(self, df, split):
        for vec_method in self.vectorizers:
            self.current_iteration["vectorizer"] = VECTORIZER_METHODS[vec_method]()
            self.row_info["vectorizer"] = vec_method
            
            for pt_method in self.problem_transformations:
                multi_model, preprocessing = PROBLEM_TRANSFORM_METHODS[pt_method].values()
                preprocessing = preprocessing()
                
                self.current_iteration["multi_model"] = multi_model
                self.current_iteration["preprocessing"] = preprocessing
                self.row_info["pt_method"] = pt_method
                
                for model in self.models:
                    base_model, model_params = AVAIBLE_MODELS[ str(model) ]
                    base_model = base_model(**model_params)
                    
                    self.current_iteration["model"] = base_model
                    self.row_info["model"] = model
                    
                    labels_col = df[self.label_col_name]
                    preprocessing.fit(labels_col)
                    
                self.run_augmentation(split)
        
    def run_augmentation(self, split):
        
        X_train = split[0]
        delta_steps = self.calc_delta_steps( X_train.shape[0] )
        
        for aug_method in self.aug_methods.methods:
            self.current_iteration["aug_method"] = aug_method
            self.row_info["aug_method"] = aug_method['label']
            for aug_selection_method in self.aug_methods.select_input_method:
                self.row_info["aug_choice_method"] = aug_selection_method
                
                X_train, _, y_train, _ = split
                df_to_aug = pd.DataFrame({"text_column": X_train, "labels_column": y_train} )
                
                self.current_iteration["aug_steps"] = np.sort( self.aug_methods.steps )
                self.current_iteration["df_to_aug"] = df_to_aug
                
                for k, step in enumerate(delta_steps):
                    self.augmentation_iteration(k, step, split)
                    self.iter_count += 1
                    
                self.results.append(self.row_info)
                self.update_fn( self.iter_count / self.total_iterations, self.get_update_text() )
                    
                    
    def augmentation_iteration(self, k, step, split):
        time_performance_start = time.time()
        
        _, X_test, _, y_test = split
        
        aug_method = self.current_iteration["aug_method"]["method"]
        aug_method.calc_n_samples = lambda _: step
        
        aug_choice = self.row_info["aug_choice_method"]
        df_to_aug, _ = AugmentDataframe( self.current_iteration["df_to_aug"], "text_column", "labels_column", aug_method, aug_choice )
        
        base_model, model_params = AVAIBLE_MODELS[ self.row_info["model"] ]
        base_model = base_model(**model_params)
        
        X_aug = df_to_aug["text_column"]
        y_aug = df_to_aug["labels_column"]

        vectorizer = self.current_iteration["vectorizer"]
        multi_model = self.current_iteration["multi_model"]
        preprocessing = self.current_iteration["preprocessing"]
        aug_clf = train_model(vectorizer, multi_model, preprocessing, base_model, X_aug, y_aug )
        
        aug_kind = self.aug_methods.aug_kind
        aug_steps = self.current_iteration["aug_steps"]
        suffix = '%' if aug_kind == 'ratio' else ''
        step_label = aug_steps[k-1] if k > 0 else "base"
        prefix = f"{'+' if k > 0 else ''}{step_label}{suffix}"
        
        aug_performance = get_performance( aug_clf, preprocessing, X_test, y_test, prefix=prefix, round_=4, percentage=True )
        
        self.row_info = {
            **self.row_info,
            f"{prefix}_train_samples": X_aug.shape[0],
            **aug_performance,
            f"{prefix}_time_performace": time.time() - time_performance_start
        }
            
    def get_update_text(self):
        dataset = self.row_info["df"]
        vectorizer = self.row_info["vectorizer"]
        model = self.row_info["model"]
        aug_method = self.row_info["aug_method"]
        aug_choice_method = self.row_info["aug_choice_method"]
        
        percentage = 100 * self.iter_count / self.total_iterations
        if percentage == 100:
            return f"Pipeline completed"
        
        return f"{percentage:.2f}% - {dataset}.{vectorizer}.{model}.{aug_method}.{aug_choice_method}"
            
    def calc_delta_steps(self, n_samples):
        aug_kind = self.aug_methods.aug_kind
        aug_steps = np.sort( self.aug_methods.steps )
        
        delta_steps = []
        if aug_kind == "count":
            delta_steps = [ 0, aug_steps[0] ]
            for j in range(1, len(aug_steps)):
                value = aug_steps[j]
                delta_steps.append( value - aug_steps[j-1] )
                
        elif aug_kind == "ratio":
            to_ratio = lambda x: int(x/100 * n_samples ) 
            delta_steps = [ 0, to_ratio(aug_steps[0])]
            for j in range(1, len(aug_steps)):
                value = aug_steps[j]
                delta_steps.append( int( to_ratio(value) - sum(delta_steps) )  )
                
        return delta_steps
        
    def calc_iterations(self):
        sampling_count = sum([method.n_iterations for method in self.resampling_methods])
    
        total_cases = len(self.datasets)
        total_cases *= sampling_count
        total_cases *= len(self.vectorizers) * len(self.problem_transformations) * len(self.models)
        
        aug_selection = self.aug_methods.select_input_method 
        aug_steps = len(self.aug_methods.steps) + 1 
        total_cases *= len(self.aug_methods.methods) * len(aug_selection) * aug_steps
      
        return total_cases