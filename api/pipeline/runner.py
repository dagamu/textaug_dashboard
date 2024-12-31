import numpy as np
import pandas as pd

import time

from sklearn.preprocessing import MultiLabelBinarizer
from src.lwise_performance import get_performance

class PipelineRunner:
    
    def __init__(self, session ) -> None:
        self.session = session
        self.mlb = MultiLabelBinarizer()
        
    def run(self, update_fn):
        
        self.results = []
        self.row_info = {}
        self.current_iteration = {}
        self.update_fn = update_fn
        
        self.datasets = self.session.datasets.items
        self.sampling_methods = self.session.sampling.items
        
        self.iter_count = 0
        self.total_iterations = self.calc_iterations()
        
        for dataset in self.datasets:
            self.row_info = {}
            
            if not dataset.loaded:
                dataset.get_data()
            
            self.current_iteration["dataset"] = dataset
            self.row_info["dataset"] = dataset.name
            
            for sampler in self.sampling_methods:    
                # TODO: PREPROCESSING   
                if dataset.params["y_format"] != 'BINARY':
                    dataset.y_train = self.mlb.fit_transform( dataset.y_train )
                    dataset.y_test = self.mlb.fit_transform( dataset.y_test )
                dataset.params["y_format"] = 'BINARY'
                
                dataset_samples = sampler.get_samples(dataset)
                self.run_subset(dataset_samples)
                
    def run_subset(self, dataset_samples):
        for df_mask in dataset_samples:
            self.current_iteration["sample_mask"] = df_mask.astype(bool)            
            self.row_info["n_samples"] = np.sum(df_mask)
            self.run_training()
            
    def run_training(self):
        
        clf_manager = self.session.classification
        text_rp_methods =  clf_manager.selected_methods["text_representation"]
        clf_models = clf_manager.selected_methods["classification_model"]
        pt_methods = clf_manager.selected_methods["problem_transformation"]
        
        for text_rp in text_rp_methods:
            self.current_iteration["text_rp"] = text_rp
            self.row_info["text_rp"] = text_rp.key
            
            for pt_method in pt_methods:
                
                self.current_iteration["multi_model"] = pt_method
                self.row_info["pt_method"] = pt_method.key
                
                for model in clf_models:
                    base_model = model
                    self.current_iteration["model"] = base_model
                    self.row_info["model"] = base_model.key
                    
                self.run_augmentation()
        
    def run_augmentation(self):
        
        aug_manager = self.session.aug_manager
        aug_selection_methods = aug_manager.aug_selection_methods
        aug_methods = aug_manager.items
        
        delta_steps = self.calc_delta_steps()
        self.current_iteration['delta_steps'] = delta_steps
        
        for aug_method in aug_methods:
            self.current_iteration["aug_method"] = aug_method
            self.row_info["aug_method"] = aug_method.key
            
            for aug_selection in aug_selection_methods:
                self.current_iteration["aug_selection"] = aug_selection()
                self.row_info["aug_selection"] = aug_selection.key
                
                dataset = self.current_iteration['dataset']
                dataset.X_aug = []
                dataset.y_aug = []
                
                for k, step in enumerate(delta_steps):
                    self.augmentation_iteration(dataset, k, step)
                    self.iter_count += 1
                    self.update_fn( self.iter_count / self.total_iterations, self.get_update_text() )
                    
                self.results.append(dict(self.row_info))
                    
                    
    def augmentation_iteration(self, dataset, k, step):
        time_performance_start = time.time()
        
        aug_method = self.current_iteration["aug_method"]
        aug_selection = self.current_iteration["aug_selection"]
        
        aug_manager = self.session.aug_manager
        clf_manager = self.session.classification
        
        if step > 0:
            aug_manager.AugmentDataset(dataset, step, aug_method, aug_selection)
        
        base_model = self.current_iteration["model"].base
        text_rp = self.current_iteration["text_rp"].base
        multi_model = self.current_iteration["multi_model"].base
        #preprocessing = self.current_iteration["preprocessing"]
        
        mask = self.current_iteration["sample_mask"]
        X_trainf, y_trainf = dataset.full_train_data(mask)
        
        aug_clf = clf_manager.train_model(text_rp, multi_model, base_model, X_trainf, y_trainf )
        
        aug_kind = self.current_iteration['aug_kind']
        aug_steps = self.current_iteration["aug_steps"]
        suffix = '%' if aug_kind == '%' and k > 0 else ''
        step_label = aug_steps[k-1] * (100 if aug_kind == '%' else 1) if k > 0 else "base"
        prefix = f"{'+' if k > 0 else ''}{step_label}{suffix}"
        
        aug_performance = get_performance( aug_clf, dataset.X_test, dataset.y_test, prefix=prefix, round_=4, percentage=True )
        
        self.row_info = {
            **self.row_info,
            f"{prefix}_train_samples": len(X_trainf),
            **aug_performance,
            f"{prefix}_time_performace": time.time() - time_performance_start
        }
            
    def get_update_text(self):
        dataset = self.row_info["dataset"]
        vectorizer = self.row_info["text_rp"]
        model = self.row_info["model"]
        aug_method = self.row_info["aug_method"]
        aug_choice_method = self.row_info["aug_selection"]
        
        percentage = 100 * self.iter_count / self.total_iterations
        
        if percentage == 100:
            return f"Pipeline completed"
        
        return f"{percentage:.2f}% - {dataset}.{vectorizer}.{model}.{aug_method}.{aug_choice_method}"
            
    def calc_delta_steps(self):
        aug_steps = np.sort( self.session.aug_manager.steps )
        aug_kind = "N" if np.all(aug_steps % 1 == 0) else "%"
        self.current_iteration['aug_kind'] = aug_kind
        self.current_iteration['aug_steps'] = aug_steps
        
        train_size = self.row_info["n_samples"]
        
        delta_steps = []
        if aug_kind == "N":
            delta_steps = [ 0, aug_steps[0] ]
            for j in range(1, len(aug_steps)):
                value = aug_steps[j]
                delta_steps.append( value - aug_steps[j-1] )
                
        elif aug_kind == "%":
            to_ratio = lambda x: int(x * train_size) 
            delta_steps = [ 0, to_ratio(aug_steps[0]) ]
            for j in range(1, len(aug_steps)):
                value = aug_steps[j]
                delta_steps.append( int( to_ratio(value) - sum(delta_steps) )  )
                
        return delta_steps
        
    def calc_iterations(self):
        sampling_count = sum([method.n_iterations for method in self.sampling_methods])
    
        clf_manager = self.session.classification
        text_rp_methods =  clf_manager.selected_methods["text_representation"]
        clf_models = clf_manager.selected_methods["classification_model"]
        pt_methods = clf_manager.selected_methods["problem_transformation"]
    
        total_cases = len(self.datasets)
        total_cases *= sampling_count
        total_cases *= len(text_rp_methods) * len(pt_methods) * len(clf_models)
        
        aug_selection = self.session.aug_manager.aug_selection_methods 
        aug_steps = len(self.session.aug_manager.steps) + 1 
        total_cases *= len(self.session.aug_manager.items) * len(aug_selection) * aug_steps
      
        return total_cases