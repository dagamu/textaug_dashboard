import streamlit as st
import numpy as np

from api.genetic_sampler_api import GeneticSamplerAPI
from utils import custom_multiselect_old

class FullDatasetSampler:
    name = "Full Dataset"
    n_iterations = 1
    
    def render(self):
        st.markdown( f"**{self.name}**" )
        
    def get_sample(self, y):
        return [np.ones(y.shape[0])]

class RandomSampler:
    name = "Random Sampler"
    
    def render(self):
        st.markdown( f"**{self.name}**" )
        
        self.n_samples = custom_multiselect_old("Select N° Samples", [100], [100], 100, key="RandomSamplerN" )
        self.n_iterations = len(self.n_samples)
        
    def get_sample(self, y):
        
        result = []
        for n_samples in self.n_samples:
            mask = np.ones(y.shape[0])
            replace = y.shape[0] < n_samples
            ones_indices = np.random.choice( y.shape[0], n_samples, replace=replace )
            mask[ones_indices] = 1
            result.append(mask)
            
        return result

class ResamplingList:
    methods = []
        
    aviable_methods = [FullDatasetSampler(), RandomSampler(), GeneticSamplerAPI()]
    default_method = aviable_methods[0]
    
    def check_methods(self):
        if len(self.methods) == 0:
            self.methods = [self.default_method]
        
    def render(self):
        st.markdown("**Dataset Re-sampling**")
        
        self.methods = st.multiselect("Add Re-sampling Method", self.aviable_methods, self.default_method, format_func=lambda method: method.name, on_change=self.check_methods )
        
        with st.container(border=True):
            for method in self.methods:
                method.render()
                st.divider()